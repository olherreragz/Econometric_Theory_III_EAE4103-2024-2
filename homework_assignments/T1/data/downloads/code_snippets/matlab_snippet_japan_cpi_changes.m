urls=["https://stats.bis.org/api/v2/data/dataflow/BIS/WS_LONG_CPI/1.0/M.JP.771?format=csv"];

for i = 1:length(urls)
  data(i)= webread(string(urls(i)));
end