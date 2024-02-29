predict_function = function(G.test,map.test, ID_list_dir=NULL, material_save_dir=NULL, env_name, result_dir=NULL, Ncores){
  if(is.null(material_save_dir)){
    material_save_dir <- getwd()
  }else{
    if(!dir.exists(material_save_dir)){
      stop("'material_save_dir' is the same directory as 'material_out_dir' used in training.")
    }
  }
  if(is.null(result_dir)){
    result_dir <- getwd()
  }else{
    if(!dir.exists(result_dir)){
      paste("path:'", result_dir, "' doesn't exist.",sep = '')
    }
  }
  reticulate::use_condaenv(env_name)
  reticulate::py_config()
  deep_map <- fread(paste(material_save_dir,'deep_map.txt',sep='/'),header = TRUE)
  snp_match_result <- snp_match(deep_map,map.test)
  prs_array <- matrix(rep(0),nrow(G.test),22)
  prs_bias <- fread(paste(material_save_dir,'prs_bias.txt',sep='/'))
  message('calculating 22 prs...')
  system.time({
    for(chr in 1:22){
      ind.chr = which(snp_match_result$chr == chr) 
      prs_array[,chr] <- prs_bias$V1[chr] + big_prodVec(G.test,snp_match_result$beta[ind.chr],ind.col = snp_match_result$`_NUM_ID_`[ind.chr],ncores = Ncores)
    }
  })
  fwrite(prs_array,paste(material_save_dir,'prs22.csv',sep='/'),sep=',',col.names = FALSE,row.names = FALSE,quote=FALSE)
  py$material_dir = material_save_dir
  py$result_dir = result_dir
  source_python('Predict/predict.py')
  pred_result <- fread(paste(result_dir,'pred.txt',sep = '/'),header=FALSE)
  if(is.null(ID_list_dir)){
    final_pred <- data.frame(ID=FAM.test,pred = pred_result)
  }else{
    ID_list <- fread(ID_list_dir,header=TRUE)
    final_pred <- data.frame(ID=ID_list$ID,pred = pred_result)
  }
  names(final_pred) <- c('ID', 'pred')
  fwrite(final_pred,paste(result_dir,'pred.txt',sep = '/'),col.names = TRUE,row.names = FALSE,quote = FALSE,sep='\t')
  message(paste('The prediction output is stored in ',result_dir,'pred.txt',sep = ''))
  return(final_pred)
  
}




