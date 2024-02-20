library(bigsnpr)
library(data.table)
library(dplyr)
library(doParallel)
library(reticulate)





read_file <- function(input_file_dir, plink_dir = NULL, bfile_out_dir = NULL) {
  # Verify if the input file exists.
  if (!(all(file.exists(paste(input_file_dir, c('.bed', '.bim', '.fam'), sep = ''))) || file.exists(input_file_dir))) {
    stop('The input file does not exist.')
  } else if (tools::file_ext(input_file_dir) == 'vcf') {
    # Convert the vcf file format to the plink file format.
    if (is.null(plink_dir)) {
      # If the PLINK directory is not provided, download the plink under the same directory as the input file.
      plink_dir <- paste(dirname(input_file_dir), '/plink', sep = '')
      download_plink_file <- if (!file.exists(plink_dir)) {
        message('PLINK is currently downloading...\n')
        download_plink(dir = dirname(input_file_dir))
        plink_dir
      } else plink_dir
      bfile_out_dir <- if (is.null(bfile_out_dir)) dirname(input_file_dir) else bfile_out_dir
      if (!file.exists(bfile_out_dir)) {
        stop("'Bfile_out_dir' does not exist.")
      }
      message(paste("Backup bfiles are in '", bfile_out_dir, "/'.", sep=''))
    } else {
      # # Verify if plink exists.
      if (!file.exists(plink_dir)) {
        stop('PLINK not found.')
      } else {
        # If the bfile_out_dir is not provided, bfile(.bed .bim .fam) will be generated under the same directory as the input file.
        bfile_out_dir <- if (is.null(bfile_out_dir)) dirname(input_file_dir) else bfile_out_dir
        if (!file.exists(bfile_out_dir)) {
          stop("'Bfile_out_dir' does not exist.")
        }
        message(paste("Backup bfiles are in '", bfile_out_dir, "/'.", sep=''))
      }
    }
    file_name <- tools::file_path_sans_ext(basename(input_file_dir))
    system2(plink_dir, paste(" --vcf ", input_file_dir, " --out ", bfile_out_dir, '/', file_name, sep=''))
    file_path <- paste(bfile_out_dir, '/', file_name, sep='')
    
    # Verify if the backing file (.bk) exists.
    if (file.exists(paste(file_path, '.bk', sep=''))) {
      message('Attaching bfiles.\n')
      obj.bigSNP <- snp_attach(paste(file_path, '.rds', sep=''))
    } else {
      message('Reading bfiles and creating rds file.\n')
      snp_readBed(paste(file_path, '.bed', sep=''))
      message('Attaching bfiles.\n')
      obj.bigSNP <- snp_attach(paste(file_path, '.rds', sep=''))
    }
    message('Done.')
    return(obj.bigSNP)
  } else if (tools::file_ext(input_file_dir) == '') {
    file_name <- tools::file_path_sans_ext(basename(input_file_dir))
    dir_name <- dirname(input_file_dir)
    if (file.exists(paste(dir_name, '/', file_name, '.bk', sep=''))) {
      message('Attaching bfiles...\n')
      obj.bigSNP <- snp_attach(paste(dir_name, '/', file_name, '.rds', sep=''))
    } else {
      message('Reading bfiles and creating rds file.\n')
      snp_readBed(paste(dir_name, '/', file_name, '.bed', sep=''))
      message('Attaching bfiles...\n')
      obj.bigSNP <- snp_attach(paste(dir_name, '/', file_name, '.rds', sep=''))
    }
    message('Done.')
    return(obj.bigSNP)
  } else {
    stop("Ensure that your input is either a VCF file or PLINK files (.bed, .bim, .fam).")
  }
}



train_with_pcnn <- function(G, y_train, map, sumstats, material_out_dir, python_dir, max_evals=10, Ncores, seed){
  snp_match_result <- snp_match(sumstats, map)
  beta <- rep(NA,ncol(G))
  beta[snp_match_result$`_NUM_ID_`] <- snp_match_result$beta
  lpval <- rep(NA,ncol(G))
  lpval[snp_match_result$`_NUM_ID_`] <- -log10(as.numeric(snp_match_result$p))
  message('\n')
  message('---------Performing clumping step...---------')
  if(file.exists(paste(material_out_dir,'/all_keep.rds',sep=''))){
    all_keep <- readRDS(paste(material_out_dir,'/all_keep.rds',sep=''))
    message('all_keep is already exist.\n')
  }else{
    all_keep <- snp_grid_clumping(G, map$chr, map$pos,
                                  lpS = lpval, exclude = which(is.na(lpval)),
                                  ncores = Ncores)
    saveRDS(all_keep,paste(material_out_dir,'/all_keep.rds',sep=''))
    message('Clumping-step is complete.\n')
  }

  message('---------Calculating Multi-PRS...---------')
  if(file.exists(paste(material_out_dir,'/multi_prs_train.rds',sep=''))){
    multi_PRS <- readRDS(paste(material_out_dir,'/multi_prs_train.rds',sep=''))
    message('multi_PRS is already exist.\n')
  }else{
    multi_PRS <- snp_grid_PRS(G, all_keep, beta, lpval,
                              backingfile = paste(material_out_dir,'/multi_prs_train',sep=''),
                              n_thr_lpS = 50, ncores = Ncores)
    message('Multi-PRS calculations are complete.\n')
  }
  reticulate::use_python(python_dir)
  reticulate::py_config()
  py$seed <- seed
  py$multi_train <- multi_PRS[,]
  py$y_pheno <- y_train
  py$out_dir <- material_out_dir
  py$max_evals <- max_evals
  py$Ncores <- Ncores
  message('---------Start training...---------\n')
  source_python("Train/train.py")
  lpval_max = max(lpval,na.rm = TRUE)
  lpval_min = min(lpval,na.rm = TRUE)
  lpval_threshold <- 0.9999 * seq_log(max(0.1,lpval_min), lpval_max,50)
  prs_coef <- fread(paste(material_out_dir,'/prs_coeff.txt',sep=''),header = FALSE)
  cl <- makeCluster(Ncores)
  registerDoParallel(cl, cores=Ncores)
  message('Calculating reverse weights...\n')
  beta_chr_reverse <- foreach(i = 1:22,.combine = c) %dopar% {
    ind.chr = which(map$chr==i)
    beta_sum <- rep(0,length(ind.chr))
    for(j in 1:28){
      for(k in 1:50){
        ind_chr_keep <- na.omit(all_keep[[i]][[j]][which(lpval[all_keep[[i]][[j]]] > lpval_threshold[k])])
        beta_gwas <- rep(0,dim(G)[2])
        beta_gwas[ind_chr_keep] <- beta[ind_chr_keep]
        beta_chr <- beta_gwas[ind.chr]
        beta_sum <- beta_chr*prs_coef$V1[(1400*(i-1) + 50*(j-1) + k)] + beta_sum
      }
    }      
    return(beta_sum)
  }
  stopCluster(cl)
  map$beta <- beta_chr_reverse
  deep_map <- map[-which(map$beta==0),]
  fwrite(deep_map,paste(material_out_dir,'/deep_map.txt',sep=''),sep='\t',col.names = TRUE,row.names = FALSE,quote = FALSE)
  message('Training step complete!')
}





