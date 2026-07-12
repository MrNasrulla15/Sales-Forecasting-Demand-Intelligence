from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = DATA_DIR / "train.csv"

COL_ORDER_DATE = "Order Date"
COL_SHIP_DATE = "Ship Date"
COL_SALES = "Sales"
COL_REGION = "Region"
COL_CATEGORY = "Category"
COL_SUBCATEGORY = "Sub-Category"
COL_ORDER_ID = "Order ID"
COL_PRODUCT_ID = "Product ID"
COL_PRODUCT_NAME = "Product Name"
COL_YEAR = "Year"
COL_MONTH = "Month"
COL_MONTH_NAME = "MonthName"
COL_WEEK = "Week"
COL_DAY_OF_WEEK = "DayOfWeek"
COL_QUARTER = "Quarter"
COL_SEASON = "Season"
COL_SHIP_DELAY_DAYS = "ShipDelayDays"
RANDOM_STATE = 42

COLOR_PALETTE = {
    "actual": "#1f77b4",
    "forecast": "#ff7f0e",
    "anomaly": "#d62728",
    "ci_band": "#fdd0a2",
}
