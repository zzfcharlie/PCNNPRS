predict_function = function(G.test,map.test, ID_list_dir, material_out_dir, python_dir, result_dir, Ncores){
  reticulate::use_python(python_dir)
  reticulate::py_config()
  deep_map <- fread(paste(material_out_dir,'/deep_map.txt',sep=''),header = TRUE)
  snp_match_result <- snp_match(deep_map,map.test)
  prs_array <- matrix(rep(0),nrow(G.test),22)
  prs_bias <- fread(paste(material_out_dir,'/prs_bias.txt',sep=''))
  message('calculating 22 prs...')
  system.time({
    for(chr in 1:22){
      ind.chr = which(snp_match_result$chr == chr) 
      prs_array[,chr] <- prs_bias$V1[chr] + big_prodVec(G.test,snp_match_result$beta[ind.chr],ind.col = snp_match_result$`_NUM_ID_`[ind.chr],ncores = Ncores)
    }
  })
  fwrite(prs_array,paste(material_out_dir,'/prs22.csv',sep=''),sep=',',col.names = FALSE,row.names = FALSE,quote=FALSE)
  py$material_dir = material_out_dir
  py$result_dir = result_dir
  source_python('Predict/predict.py')
  pred_result <- fread(paste(result_dir,'pred.txt',sep = ''),header=FALSE)
  ID_list <- fread(ID_list_dir,header=TRUE)
  final_pred <- data.frame(ID=ID_list$ID,pred = pred_result)
  names(final_pred) <- c('ID', 'pred')
  fwrite(final_pred,paste(result_dir,'pred.txt',sep = ''),col.names = TRUE,row.names = FALSE,quote = FALSE,sep='\t')
  return(final_pred)
  message(paste('The prediction output is stored in ',result_dir,'pred.txt',sep = ''))
}




