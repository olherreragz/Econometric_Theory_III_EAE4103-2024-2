import pandas as pd

urls = ["https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/M.N.B.JP?format=csv"]

df = pd.concat([pd.read_csv(url) for url in urls])

