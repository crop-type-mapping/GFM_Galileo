library(terra)
district = 'Nyagatare'
season = 'B'
eyear = '2025'
path <- '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/data/outputs/'
root <- '/home/bkenduiywo/GFM_Galileo/results/'
labels <- rast(paste0(path,district,'_',season,eyear,'_merged_masked_labels.tif'))

NAflag(labels) <- 255
# Increment all non-NA values by 1
#labels[!is.na(labels)] <- labels[!is.na(labels)] + 1
freq(labels)
classnames <- c('Bean', 'Irish Potato', 'Maize', 'Rice')  # 0-3 and 255
class_colors <- c("#55FF00","#732600", "#FFD400" , "#00A9E6")  # match exactly
# Display function


display <- function(map, method){
  par(mar = c(7, 2, 1.6, 6)) #c(bottom, left, top, right)
  image(map, col = class_colors, axes = TRUE, ann = FALSE)
  #add_legend("bottomright", legend = classnames, fill = class_colors, ncol = 3, bty = 'n', cex = 1.1)
  legend("topright", legend = classnames, fill = class_colors, ncol = 1, bty = 'n', cex = 1.1)
  title(paste0(method, " classification"))
}
output_file <- paste0(root,district,'_',season,eyear,'_merged_labels.png')
png(filename = output_file, width = 8.2, height = 6.6, units = "in", res = 300)
display(labels, "Finetuned Galileo GFM")
dev.off()


