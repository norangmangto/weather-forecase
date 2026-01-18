import os
import requests
import pandas as pd
import duckdb
import io
import zipfile
from datetime import datetime, timedelta

class WeatherIngestor:
    def __init__(self, db_path="data/weather.db"):
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path)

    def ingest_dwd_historical(self, station_id="01078"):
        """
        Ingests historical daily climate data (KL) from DWD.
        Station 01078 is Düsseldorf.
        """
        print(f"Fetching DWD historical data for station {station_id}...")
        base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/"

        # In a real scenario, we'd list the directory to find the exact filename.
        # For Düsseldorf 01078, we know it's roughly tageswerte_KL_01078_19520101_20241231_hist.zip (dates change)
        # We'll use a simple scraper to find the file.
        from bs4 import BeautifulSoup

        r = requests.get(base_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        filename = None
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and f"_{station_id}_" in href and href.endswith(".zip"):
                filename = href
                break

        if not filename:
            print(f"Could not find historical file for station {station_id}")
            return

        file_url = base_url + filename
        print(f"Downloading from {file_url}...")
        r = requests.get(file_url)

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # Find the actual data file (starts with 'produkt_klima_tag_')
            data_files = [f for f in z.namelist() if f.startswith('produkt_klima_tag_')]
            if not data_files:
                print(f"Could not find data file in {filename}. Available files: {z.namelist()}")
                return

            data_filename = data_files[0]
            print(f"Extracting {data_filename}...")
            with z.open(data_filename) as f:
                df = pd.read_csv(f, sep=';', na_values=[-999.0])
                # Strip whitespace from columns
                df.columns = [c.strip() for c in df.columns]

        # Register in DuckDB
        self.conn.execute("CREATE OR REPLACE TABLE raw_dwd_kl AS SELECT * FROM df")
        print(f"Ingested {len(df)} rows into raw_dwd_kl")

    def ingest_dwd_recent(self, station_id="01078"):
        """
        Ingests recent daily climate data (KL) from DWD (gap filling).
        """
        print(f"Fetching DWD recent data for station {station_id}...")
        base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/recent/"
        filename = f"tageswerte_KL_{station_id}_akt.zip"

        file_url = base_url + filename
        print(f"Downloading from {file_url}...")

        try:
            r = requests.get(file_url)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to download recent data: {e}")
            return

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            data_files = [f for f in z.namelist() if f.startswith('produkt_klima_tag_')]
            if not data_files:
                print(f"Could not find data file in {filename}.")
                return

            data_filename = data_files[0]
            print(f"Extracting {data_filename}...")
            with z.open(data_filename) as f:
                df = pd.read_csv(f, sep=';', na_values=[-999.0])
                df.columns = [c.strip() for c in df.columns]

        # Append to existing table
        print(f"Ingested {len(df)} rows of recent data")
        self.conn.execute("INSERT INTO raw_dwd_kl SELECT * FROM df")

        # Deduplicate
        print("Deduplicating raw_dwd_kl...")
        self.conn.execute("""
            CREATE TABLE raw_dwd_kl_dedup AS
            SELECT DISTINCT * FROM raw_dwd_kl
        """)
        self.conn.execute("DROP TABLE raw_dwd_kl")
        self.conn.execute("ALTER TABLE raw_dwd_kl_dedup RENAME TO raw_dwd_kl")
        print("Deduplication complete.")

    def ingest_openmeteo_air_quality(self, lat=51.2217, lon=6.7762):
        """
        Ingests historical air quality (fine dust) from Open-Meteo.
        Düsseldorf coordinates approx.
        """
        print("Fetching Open-Meteo air quality data...")
        # Historical air quality API
        start_date = "2020-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        pollutants = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly={pollutants}&start_date={start_date}&end_date={end_date}&format=json"

        r = requests.get(url)
        data = r.json()

        if "hourly" not in data:
            print("No hourly data found in Open-Meteo response")
            return

        df = pd.DataFrame(data["hourly"])
        df['time'] = pd.to_datetime(df['time'])

        # Aggregate to daily (mean)
        agg_map = {p: 'mean' for p in pollutants.split(',')}
        df_daily = df.groupby(df['time'].dt.date).agg(agg_map).reset_index()
        df_daily.rename(columns={'time': 'date'}, inplace=True)
        # Rename columns to match conventions
        df_daily.columns = ['date'] + [f"{p}_mean" for p in pollutants.split(',')]

        self.conn.execute("CREATE OR REPLACE TABLE raw_openmeteo_aq AS SELECT * FROM df_daily")
        print(f"Ingested {len(df_daily)} rows into raw_openmeteo_aq")

    def ingest_openmeteo_supplemental(self, lat=51.2217, lon=6.7762):
        """
        Ingests supplemental weather data (UV Index, Visibility) from Open-Meteo Historical API.
        """
        print("Fetching Open-Meteo supplemental weather data...")
        start_date = "2020-01-01"
        # Archive API usually lags by 1 day
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=uv_index,visibility&format=json"

        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"Failed to fetch supplemental data: {e}")
            # Ensure table exists even if empty
            self.conn.execute("CREATE TABLE IF NOT EXISTS raw_openmeteo_supplement (date DATE, uv_index_max DOUBLE, visibility_mean DOUBLE)")
            return

        if "hourly" not in data:
            print("No hourly data found in Open-Meteo supplemental response")
            self.conn.execute("CREATE TABLE IF NOT EXISTS raw_openmeteo_supplement (date DATE, uv_index_max DOUBLE, visibility_mean DOUBLE)")
            return

        df = pd.DataFrame(data["hourly"])
        df['time'] = pd.to_datetime(df['time'])

        # Aggregate to daily (mean)
        df_daily = df.groupby(df['time'].dt.date).agg({
            'uv_index': 'max', # Max UV index of the day is usually more relevant
            'visibility': 'mean'
        }).reset_index()
        df_daily.columns = ['date', 'uv_index_max', 'visibility_mean']

        self.conn.execute("CREATE OR REPLACE TABLE raw_openmeteo_supplement AS SELECT * FROM df_daily")
        print(f"Ingested {len(df_daily)} rows into raw_openmeteo_supplement")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    ingestor = WeatherIngestor()
    ingestor.ingest_dwd_historical()
    ingestor.ingest_dwd_recent()
    ingestor.ingest_openmeteo_air_quality()
    ingestor.ingest_openmeteo_supplemental()
    ingestor.close()
