import tempfile
import zipfile
import duckdb
from pathlib import Path


class FinancialDataIngester:
    def __init__(self, db_path:str):
        self.db_path = db_path
        self.con = duckdb.connect(self.db_path)

    def ingest_zip(self, zip_path:str, table_name:str):
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file=zip_path, mode="r") as zip_ref:
                zip_ref.extractall(temp_dir)
            csv_files = str(Path(temp_dir) / "**" / "*.csv")

            self.con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS \
                SELECT *, regexp_extract(filename, '([^/\\\\]+)$') as source_file \
                FROM read_csv_auto('{csv_files}', filename=true, delim=',', union_by_name=true, sample_size=-1)")

    def close(self):
        self.con.close()

ingester = FinancialDataIngester(db_path="financial_data.db")
ingester.ingest_zip(zip_path="archive.zip", table_name="nifty500")
ingester.close()
