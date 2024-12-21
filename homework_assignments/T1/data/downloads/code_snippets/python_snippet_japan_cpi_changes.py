import pandas as pd

urls = ["https://stats.bis.org/api/v2/data/dataflow/BIS/WS_LONG_CPI/1.0/M.JP.771?format=csv"]

df = pd.concat([pd.read_csv(url) for url in urls])
