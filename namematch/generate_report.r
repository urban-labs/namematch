
library(yaml)

args = commandArgs(trailingOnly = TRUE)

# knit the report
rmarkdown::render(
  input = args[3],
  params = list(stats_file = args[1], output_dir = args[2]),
  output_file = args[4]
)
