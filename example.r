{
  source('Train/train_with_pc.r')
  source('Predict/predict_with_pc.R')
}



{
  obj.bigSNP <- read_file('data/train')
  G <- obj.bigSNP$genotypes
  map <- obj.bigSNP$map[-3]
  sumstats <- fread('data/sumstats.txt',sep='\t')
  names(map) <- c('chr', 'rsid', 'pos', 'a1','a0')
  names(sumstats) <- c('chr', 'rsid', 'pos', 'a1', 'a0', 'beta', 'p')
  y_train <- fread('data/y_train.txt', header=FALSE)
}



train_with_pcnn(
    G,
    y_train,
    map,
    sumstats,
    env_name = 'yourenv',
    material_out_dir = 'material_out/',
    max_evals = 20,
    Ncores = nb_cores(),
    seed = 32)


{
  obj.bigSNP.test <- read_file("data/test")
  G.test   <- obj.bigSNP.test$genotypes
  CHR.test <- obj.bigSNP.test$map$chromosome
  POS.test <- obj.bigSNP.test$map$physical.pos
  FAM.test<-obj.bigSNP.test$fam$family.ID
  map.test <- obj.bigSNP.test$map[,-3]
  names(map.test) <- c('chr','rs','pos','a1','a0')
}


result <- predict_function(G.test,
                           map.test,
                           env_name = 'yourenv',
                           material_save_dir = 'material_out/',
                           result_dir = 'result/',
                           Ncores = nb_cores())



