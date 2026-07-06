import duckdb
from fastmcp import FastMCP
import financial_tools as tools
import pathlib

mcp = FastMCP("Financial-Analyst-Server")
DB_PATH = str(pathlib.Path(__file__).parent.parent / "data_source" / "financial_data.db")

@mcp.tool()
def get_db_schema() -> str:
    """Get the schema (column names and types) of the nifty500 table in the financial database."""
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        schema_info = con.execute("PRAGMA table_info('nifty500');").df()
        con.close()
        return f"The 'nifty500' table has the following columns:\n{schema_info.to_markdown(index=False)}"
    except Exception as e:
        return f"Error getting schema: {e}. Has the database been created with data_ingestion.py?"

@mcp.tool()
def query_db(sql_query: str) -> str:
    """Run a read-only SQL query against the financial database and return results as a markdown table."""
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        result_df = con.execute(sql_query).df()
        con.close()
        return result_df.to_markdown(index=False)
    except Exception as e:
        return f"SQL Error: {e}"

@mcp.tool()
def compute_cagr(start_val: float, end_val: float, periods: int) -> str:
    """Calculate the Compound Annual Growth Rate (CAGR) given start value, end value, and number of periods."""
    result = tools.calculate_cagr(start_val, end_val, periods)
    return f"Calculated CAGR over {periods} periods is: {result:.2%}"

@mcp.tool()
def compute_volatility(prices: list[float]) -> str:
    """Calculate the annualized volatility given a list of historical prices."""
    result = tools.calculate_volatility(prices)
    return f"The annualized volatility is: {result:.2%}"

if __name__ == "__main__":
    mcp.run(show_banner=False)

