urls=["https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/M.N.B.JP?format=csv"];

for i = 1:length(urls)
  data(i)= webread(string(urls(i)));
end

