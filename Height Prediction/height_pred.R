height_predict <- function(G,map,gender_ID_list_dir,match_by_pos=TRUE,material_dir,python_dir,result_dir,Ncores){
  reticulate::use_python(python_dir)
  reticulate::py_config()
  gender_list <- fread(gender_ID_list_dir,header = TRUE)
  ##############
  if(!all(names(gender_list)==c('ID', 'gender'))){
    stop("Make sure the names of file are 'ID' and 'gender'.")
  }else{
    ind_male.test<-which(gender_list$gender==1)
    ind_female.test<-which(gender_list$gender==2)
  }
  
  cat('loading_linear_sumstat...\n')
  linear_male_model <- fread(paste(material_dir,'/height_linear_male_model.txt',sep=''),header = TRUE)
  linear_female_model <- fread(paste(material_dir,'/height_linear_female_model.txt',sep=''),header = TRUE)
  cat('loading_complete!\n')
  ###Male has 2 models(ALL&BBJ) and Female has 3 models(ALL&ASIA&BBJ)###
  cat('loading_nonlinear_sumstat...\n')
  #male#
  non_linear_male_all <- fread(paste(material_dir,'/male_all/deep_map_male_all.txt',sep=''),header=TRUE)
  non_linear_male_bbj <- fread(paste(material_dir,'/male_bbj/deep_map_male_bbj.txt',sep=''),header=TRUE)
  
  #female#
  non_linear_female_asia <- fread(paste(material_dir,'/female_asia/deep_map_female_asia.txt',sep=''),header=TRUE)
  non_linear_female_bbj <- fread(paste(material_dir,'/female_bbj/deep_map_female_bbj.txt',sep=''),header=TRUE)
  cat('loading_complete!\n')
  
  
  
  # Define a function to match variants and print results
  match_and_print <- function(sumstats, label) {
    message(paste('Variants are matching with', label, 'sumstats.'))
    # Perform variant matching
    snp_match_result <- snp_match(sumstats = sumstats, info_snp = map, match.min.prop = 0.45,join_by_pos = match_by_pos)
    # Calculate coverage rate
    coverage_rate <- round(nrow(snp_match_result) / nrow(sumstats), 3)
    # Output messages based on coverage rate
    if (coverage_rate < 0.45) {
      stop(paste("The coverage rate is larger than 0.5", sep = ' '))
    } else if (coverage_rate >= 0.45 & coverage_rate <= 0.8) {
      message(paste('The coverage rate is ', coverage_rate, ', the final result may not be entirely accurate.', sep = ''))
    } else {
      message(paste('The coverage rate is ', coverage_rate, ".", sep = ''))
    }
    return(snp_match_result)
    message('Done!\n')
  }
  # Call the function with different datasets and labels
  snp_male_match.linear <- match_and_print(linear_male_model, 'male.linear')
  snp_female_match.linear <- match_and_print(linear_female_model, 'female.linear')
  snp_male_match.all <- match_and_print(non_linear_male_all, 'male.all')
  snp_male_match.bbj <- match_and_print(non_linear_male_bbj, 'male.bbj')
  snp_female_match.asia <- match_and_print(non_linear_female_asia, 'female.asia')
  snp_female_match.bbj <- match_and_print(non_linear_female_bbj, 'female.bbj')
  
  ##############predict##############
  pred_linear <- rep(NA,nrow(gender_list))
  pred_intercept <- rep(NA,nrow(gender_list))
  male_intercept <- as.numeric(fread(paste(material_dir,'/male_coefficients.txt',sep = ''))[2,1])
  female_intercept <- as.numeric(fread(paste(material_dir,'/female_coefficients.txt',sep = ''))[2,1])
  message('calculate_linear_results...\n')
  male_linear_part <- big_prodVec(G,snp_male_match.linear$beta,ind.col = snp_male_match.linear$`_NUM_ID_`,ind.row = ind_male.test,ncores = Ncores)
  female_linear_part <- big_prodVec(G,snp_female_match.linear$beta,ind.col = snp_female_match.linear$`_NUM_ID_`,ind.row = ind_female.test,ncores = Ncores)
  message('Done!\n')
  pred_linear[ind_male.test] <- male_linear_part
  pred_intercept[ind_male.test] <- male_intercept
  pred_linear[ind_female.test] <- female_linear_part
  pred_intercept[ind_female.test] <- female_intercept
  
  N.male <- length(ind_male.test)
  p.male <- 22
  d.male <- 2
  prs_male_array <- array(NA,dim = c(N.male,p.male,d.male))
  
  message('Calculating nonlinear 22prs...\n')
  for(chr in 1:22){
    ind.chr.all = which(snp_male_match.all$chr == chr) 
    ind.chr.bbj = which(snp_male_match.bbj$chr == chr) 
    prs_male_array[,chr,1] <- big_prodVec(G,snp_male_match.all$beta[ind.chr.all],ind.col = snp_male_match.all$`_NUM_ID_`[ind.chr.all],ind.row = ind_male.test,ncores = Ncores)
    prs_male_array[,chr,2] <- big_prodVec(G,snp_male_match.bbj$beta[ind.chr.bbj],ind.col = snp_male_match.bbj$`_NUM_ID_`[ind.chr.bbj],ind.row = ind_male.test,ncores = Ncores)
  }
  fwrite(matrix(prs_male_array[,,1],N.male,22),paste(material_dir,'/prs22_male_all.csv',sep = ''),col.names = FALSE,row.names=FALSE)
  fwrite(matrix(prs_male_array[,,2],N.male,22),paste(material_dir,'/prs22_male_bbj.csv',sep = ''),col.names = FALSE,row.names=FALSE)
  N.female <- length(ind_female.test)
  p.female <- 22
  d.female <- 2
  prs_female_array <- array(NA,dim = c(N.female,p.female,d.female))
  for(chr in 1:22){
    ind.chr.asia = which(snp_female_match.asia$chr == chr) 
    ind.chr.bbj = which(snp_female_match.bbj$chr == chr) 
    prs_female_array[,chr,1] <- big_prodVec(G,snp_female_match.asia$beta[ind.chr.asia],ind.col = snp_female_match.asia$`_NUM_ID_`[ind.chr.asia],ind.row = ind_female.test,ncores = Ncores)
    prs_female_array[,chr,2] <- big_prodVec(G,snp_female_match.bbj$beta[ind.chr.bbj],ind.col = snp_female_match.bbj$`_NUM_ID_`[ind.chr.bbj],ind.row = ind_female.test,ncores = Ncores)
  }
  fwrite(matrix(prs_female_array[,,1],N.female,22),paste(material_dir,'/prs22_female_asia.csv',sep = ''),col.names = FALSE,row.names=FALSE)
  fwrite(matrix(prs_female_array[,,2],N.female,22),paste(material_dir,'/prs22_female_bbj.csv',sep = ''),col.names = FALSE,row.names=FALSE)
  message('Done!\n')
  py_run_string(paste0("gender_ID_list_dir = '", gender_ID_list_dir, "'"))
  py_run_string(paste0("material_dir = '", material_dir, "'"))
  py_run_string(paste0("result_dir = '", result_dir, "'"))
  message('calculate_nonlinear_results...\n')
  source_python(paste(material_dir,'/predict.py',sep=''))
  message('Done!\n')
  pred_nonlinear <- fread(paste(material_dir,'/pred_nonlinear.txt',sep=''),header = FALSE)$V1
  height <- round((pred_nonlinear + pred_linear +pred_intercept),1)
  height_result <- (data.frame(ID = gender_list$ID,gender = gender_list$gender, height = as.numeric(height)))
  fwrite(height_result,paste(result_dir,'/height_result.txt',sep = ''),col.names = TRUE,row.names = FALSE,sep='\t')
  return(height_result)
}
