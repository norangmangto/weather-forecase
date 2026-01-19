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
        # Define stations: ID, Lat, Lon, Name
        self.stations = [
            {"id": "01078", "lat": 51.2217, "lon": 6.7762, "name": "Dusseldorf"},
            {"id": "00433", "lat": 52.5644, "lon": 13.3088, "name": "Berlin-Tegel"},
            {"id": "01975", "lat": 53.6332, "lon": 9.9881, "name": "Hamburg-Fuhlsbuettel"}
        ]

    def ingest_dwd_historical(self, station_id):
        """
        Ingests historical daily climate data (KL) from DWD for a given station.
        """
        print(f"Fetching DWD historical data for station {station_id}...")
        base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/"

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
            data_files = [f for f in z.namelist() if f.startswith('produkt_klima_tag_')]
            if not data_files:
                print(f"Could not find data file in {filename}.")
                return

            data_filename = data_files[0]
            print(f"Extracting {data_filename}...")
            with z.open(data_filename) as f:
                df = pd.read_csv(f, sep=';', na_values=[-999.0])
                df.columns = [c.strip() for c in df.columns]

        # Rename STATIONS_ID to station_id for consistency and ensure it's a string
        df.rename(columns={'STATIONS_ID': 'station_id'}, inplace=True)
        df['station_id'] = df['station_id'].astype(str).str.zfill(5)

        # Register in DuckDB (Append if table exists)
        table_exists = self.conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'raw_dwd_kl'").fetchone()[0] > 0
        if not table_exists:
            self.conn.execute("CREATE TABLE raw_dwd_kl AS SELECT * FROM df")
        else:
            self.conn.execute("INSERT INTO raw_dwd_kl SELECT * FROM df")

        print(f"Ingested {len(df)} rows into raw_dwd_kl for {station_id}")

    def ingest_dwd_recent(self, station_id):
        """
        Ingests recent daily climate data (KL) from DWD for a given station.
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
            print(f"Failed to download recent data for {station_id}: {e}")
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

        df.rename(columns={'STATIONS_ID': 'station_id'}, inplace=True)
        df['station_id'] = df['station_id'].astype(str).str.zfill(5)

        # Append to existing table
        print(f"Ingested {len(df)} rows of recent data for {station_id}")
        self.conn.execute("INSERT INTO raw_dwd_kl SELECT * FROM df")

    def deduplicate_dwd(self):
        """Deduplicate raw_dwd_kl table."""
        print("Deduplicating raw_dwd_kl...")
        self.conn.execute("""
            CREATE TABLE raw_dwd_kl_dedup AS
            SELECT DISTINCT * FROM raw_dwd_kl
        """)
        self.conn.execute("DROP TABLE raw_dwd_kl")
        self.conn.execute("ALTER TABLE raw_dwd_kl_dedup RENAME TO raw_dwd_kl")
        print("Deduplication complete.")

    def ingest_openmeteo_air_quality(self, station_id, lat, lon):
        """
        Ingests historical air quality from Open-Meteo for a given location.
        """
        print(f"Fetching Open-Meteo air quality data for station {station_id}...")
        start_date = "2020-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        pollutants = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly={pollutants}&start_date={start_date}&end_date={end_date}&format=json"

        r = requests.get(url)
        data = r.json()

        if "hourly" not in data:
            print(f"No hourly data found in Open-Meteo response for {station_id}")
            return

        df = pd.DataFrame(data["hourly"])
        df['time'] = pd.to_datetime(df['time'])
        df['station_id'] = station_id

        # Aggregate to daily (mean)
        agg_map = {p: 'mean' for p in pollutants.split(',')}
        df_daily = df.groupby(['station_id', df['time'].dt.date]).agg(agg_map).reset_index()
        df_daily.rename(columns={'time': 'date'}, inplace=True)
        df_daily.columns = ['station_id', 'date'] + [f"{p}_mean" for p in pollutants.split(',')]

        table_exists = self.conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'raw_openmeteo_aq'").fetchone()[0] > 0
        if not table_exists:
            self.conn.execute("CREATE TABLE raw_openmeteo_aq AS SELECT * FROM df_daily")
        else:
            self.conn.execute("INSERT INTO raw_openmeteo_aq SELECT * FROM df_daily")

        print(f"Ingested {len(df_daily)} rows into raw_openmeteo_aq for {station_id}")

    def ingest_openmeteo_supplemental(self, station_id, lat, lon):
        """
        Ingests supplemental weather data from Open-Meteo Historical API for a given location.
        """
        print(f"Fetching Open-Meteo supplemental weather data for station {station_id}...")
        start_date = "2020-01-01"
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=uv_index,visibility&format=json"

        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"Failed to fetch supplemental data for {station_id}: {e}")
            return

        if "hourly" not in data:
            print(f"No hourly data found in Open-Meteo supplemental response for {station_id}")
            return

        df = pd.DataFrame(data["hourly"])
        df['time'] = pd.to_datetime(df['time'])
        df['station_id'] = station_id

        # Aggregate to daily
        df_daily = df.groupby(['station_id', df['time'].dt.date]).agg({
            'uv_index': 'max',
            'visibility': 'mean'
        }).reset_index()
        df_daily.rename(columns={'time': 'date'}, inplace=True)
        df_daily.columns = ['station_id', 'date', 'uv_index_max', 'visibility_mean']

        table_exists = self.conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'raw_openmeteo_supplement'").fetchone()[0] > 0
        if not table_exists:
            self.conn.execute("CREATE TABLE raw_openmeteo_supplement AS SELECT * FROM df_daily")
        else:
            self.conn.execute("INSERT INTO raw_openmeteo_supplement SELECT * FROM df_daily")

        print(f"Ingested {len(df_daily)} rows into raw_openmeteo_supplement for {station_id}")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    ingestor = WeatherIngestor()

    # Clear old tables to prevent schema issues during transition
    tables = ['raw_dwd_kl', 'raw_openmeteo_aq', 'raw_openmeteo_supplement']
    for t in tables:
        ingestor.conn.execute(f"DROP TABLE IF EXISTS {t}")

    for station in ingestor.stations:
        print(f"\n--- Processing Station: {station['name']} ({station['id']}) ---")
        ingestor.ingest_dwd_historical(station['id'])
        ingestor.ingest_dwd_recent(station['id'])
        ingestor.ingest_openmeteo_air_quality(station['id'], station['lat'], station['lon'])
        ingestor.ingest_openmeteo_supplemental(station['id'], station['lat'], station['lon'])

    ingestor.deduplicate_dwd()
    ingestor.close()
