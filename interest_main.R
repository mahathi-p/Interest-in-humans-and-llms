########################################
# Predictors of HUMAN interest ratings #
########################################

# load packages
library(ggplot2)
library(lme4)
library(plotly)
library(GGally)
library(tidyverse)
library(httr)
library(doBy)
library(r2mlm)
library(kableExtra)
library(cowplot)
library(purrr)
library(broom)
library(scales)
library(irr)
library(ggpirate)

# This script will merge all trial_level data from the INTEREST project (i.e., all human annotations at the level of individual turns) and all turn-level or turn-pair-level linguistic features
# We do not deal with LLM ratings at this stage (see below for that)
# Some of the features and average human INT ratings are in this file (it also contains some of the LLM ratings, but we do not need those here)
all_feat<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/all_llms_human.csv")
# this has 6656 rows
# conversation_id
# id (Doccano-generated, should be larger for later projects and later pages within a project)

# The trial_level data are saved as separate .csv files for each project
# They are all in this directory
# https://github.com/mahathi-p/Interest-in-humans-and-llms/tree/main/data/projects
# First download the files to your local computer
# Then set the working directory to the correct folder (e.g., on my local machine)
setwd("/Users/chiara/Library/CloudStorage/OneDrive-UniversityofWarwick/rdf/data/all_label_dir")
# Then run the following to read in the files, collate and process them into the data frame all_trial

all_trial<- list.files() %>%
  as.data.frame()%>%
  pull() %>%
  map_df(~{
    .x %>%
      read_csv %>%
      rename(.,Progr_page_num_all = names(.)[1])%>% #add name for progressive page number column (first column)
      # #find all columns that contain annotations (sometimes there are 3 annotators X 2 ratings = 6, sometimes 3 annotators X 3 ratings = 9)
      # int_col_names<-names(project)[str_detect(names(project),"._int")]
      # expint_col_names<-names(project)[str_detect(names(project),"._expint")]
      # comp_col_names<-names(project)[str_detect(names(project),"._comparison")] #may be empty
      # #average across three int scores and three expint scores, but keep original scores (score 1, score 2, score 3) 
      # and create new columns for Ann1, Ann2, Ann3 Prolific Id
      mutate(Ann_1_id = gsub("_int","",names(.)[str_detect(names(.),"._int")][1]), 
             Ann_2_id = gsub("_expint","",names(.)[str_detect(names(.),"._int")][2]), 
             Ann_3_id = gsub("_comparison","",names(.)[str_detect(names(.),"._int")][3]),
             project = gsub(".csv","",.x),
             across(c(Ann_1_id,Ann_2_id, Ann_3_id),as.character))%>%
      rename_with(.,~paste0('Int_',1:3),ends_with("_int"))%>%
      rename_with(.,~paste0('ExpInt_',1:3),ends_with("_expint"))%>%
      #rename_with(.,~paste0('Comp_',1:3),ends_with("_comparison"))%>%
      #     #drop comparison scores if present
      select(!(ends_with("_comparison")))%>%
      #drop reference conversations if present
      group_by(conversation_id)%>%
      mutate(N_turns=length(Progr_page_num_all))%>%
      filter(N_turns>11)%>% #See below for a justification for using 11 as the threshold
      ungroup()%>%
      # I need to convert the conversation_id variable into numeric and strip leading zeros
      mutate(conversation_id = as.numeric(conversation_id))%>%
      rowwise()%>%
      mutate(Int_avg = mean(c(as.numeric(Int_1),as.numeric(Int_2),as.numeric(Int_3)), trim=0),
             ExpInt_avg = mean(c(as.numeric(ExpInt_1),as.numeric(ExpInt_2),as.numeric(ExpInt_3)), trim=0),
             Int_var = var(c(as.numeric(Int_1),as.numeric(Int_2),as.numeric(Int_3))),
             ExpInt_var = var(c(as.numeric(ExpInt_1),as.numeric(ExpInt_2),as.numeric(ExpInt_3))))
  })

setwd("~/Library/CloudStorage/OneDrive-UniversityofWarwick/rdf/data")

# This has now the same number of rows (6656) as all_feat
# #check distribution of conversation length (in number of pages)  
# turns_by_conv<-result %>%
#   group_by(conversation_id, project)%>%
#   summarise(N_turns=length(Progr_page_num_all))%>%
#   select(N_turns,conversation_id,project)
# 
# ggplot(turns_by_conv,aes(N_turns))+geom_histogram()
# 
## Based on the distribution, conversations that are 11 pages long are a clear outlier, 
## these must be the reference conversations to drop

# Progr_page_num_all is a continuous number across all conversations in a project
# We are not going to use it for merging. Instead we are going to use the Doccano id.
#make sure it has the same name as in all_trial
names(all_feat)[3]<-"doccano_id"

all<-merge(all_feat,all_trial,by=c("conversation_id","doccano_id"))

#check the average int values are the same
#all.equal(all$human_int_mean,all$Int_avg) #note, cannot use "==" because of this issue
#https://stackoverflow.com/questions/9508518/why-are-these-numbers-not-equal

#compute human_expint
# to do this, I need to check how human_int relates to human_int_mean (was it rounded down?)
#all$human_int_mean==all$human_int # no
#all$round_check<-round(all$human_int_mean)
#sum(all$human_int==all$round_check) #yes

all$human_expint<-round(all$ExpInt_avg)
#summary(all$human_expint)                  

#add other features
# this is the NEW gispy implementation
gispy<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/rerun_gispy_output.csv")
gispy$doccano_id<-str_replace(sapply(strsplit(gispy$d_id, "_"), "[[",2),".txt","")
gispy$conversation_id<-as.numeric(sapply(strsplit(gispy$d_id, "_"), "[[",1))
gispy<-gispy[,-(1:2)]
all<-merge(all,gispy,by=c("conversation_id","doccano_id"))
similarity<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/complete_similarity_metrics.csv", header=T)
similarity<-similarity[,c(2:3,7,10)]
names(similarity)[1]<-"doccano_id"
all<-merge(all,similarity,by=c("conversation_id","doccano_id"))
complexity<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/complete_complexity_metrics.csv",header=T)
complexity<-complexity[,c(1,2,21:34)]
names(complexity)[1]<-"doccano_id"
all<-merge(all,complexity,by=c("conversation_id","doccano_id"))

# sample code for plotting correlations
# feat_text<-all[,c("lexicon_count","gis","PCCNC_megahr","LCS", "Int_avg", "Int_var")]
# 
# corr_matrix_plot<-ggpairs(feat_text,upper = list(continuous = wrap("cor", method = "kendall")))
# ggplotly(corr_matrix_plot)

# sample code for plotting non-linear relations
# ggplot(all,aes(gis,human_int))+geom_point()+stat_smooth(method="gam")

###############################################################
### Which features are predictive of human INT and EXP INT? ###
###############################################################

# We are going to use linear mixed effects models (with random intercepts for conversation_id and annotator_id)
# and fit trial-level features (for the current turn; i.e., features of turns displayed on the same page the INT and EXP INT scores 
# were provided on Doccano),
# We are going to select the relevant features a priori. 
# First, we are going to graph/analyse correlations for each feature type separately

# Concreteness (2 measures)
# PCCNC_megahr (file: complete_gispy_scores)
# PCCNC_mrc (file: complete_gispy_scores)

# Complexity (15 measures)
# gis score (file: complete_gispy_scores)
# syllable_count (file: complete_complexity_metrics)
# lexicon_count (file: complete_complexity_metrics)
# difficult_words (file: complete_complexity_metrics)
# flesch_reading_ease	 (file: complete_complexity_metrics)
# flesch_kincaid_grade	(file: complete_complexity_metrics)
# smog_index (file: complete_complexity_metrics)	
# coleman_liau_index  (file: complete_complexity_metrics)
# automated_readability_index	 (file: complete_complexity_metrics)
# dale_chall_readability_score	(file: complete_complexity_metrics)
# spache_readability (file: complete_complexity_metrics)
# gunning_fog  (file: complete_complexity_metrics)	
# linsear_write_formula	 (file: complete_complexity_metrics)
# mcalpine_eflaw	 (file: complete_complexity_metrics)
# text_standard (file: complete_complexity_metrics)

# Uptake (5)
# We had two measures for this in a previous round of analyses:
# cos (cosine similarity between teacher and student turn on a given page, based on Glove embeddings) (file: complete_similarity_metrics)
# LCS (longest commmon subsequence between teacher and student turn on a given page) (file: complete_similarity_metrics)
# These have been added to the data frame "all" above.
# However, there may have been an error in how they were computed for turns that included a Chat-GPT generated alternative, 
# so these measures should be disregarded. 
# Instead, we use the measures included in all_uptake_final.csv
#Note: all of the measures below were computed after removing stop words and punctuation
#Note 2: sometimes there was only one speaker in a page (when that speaker started the conversation), in this case cos was set to 0
#Note 3: these have been checked for the error mentioned above (i.e., no alternative was used in the computation, just the original turns)
# cos_proc_within_page (cosine similarity between teacher and student turn on a given page, based on mixedbread-ai/mxbai-embed-large-v1 embeddings) (file: all_uptakes_final)
# LCS (longest common subsequence) (file: all_uptakes_final)
# perTinS/perSinT (number of tokens from the teacher turn that are in the student turn and vice-versa) (file: all_uptakes_final) 
# We are also using an additional measure from educonvokit, which is in this file: convokit_uptake.csv (see below for more details)

# Since for each feature type, we have multiple measures, we first use the correlations to select the best measure 
# for each feature type (i.e., the measure that has the highest correlation with INT/EXP_INT).
# NOTE: since human_int and human_expint are ordinal not continuous, we use Kendall's Tau instead of Pearson's R

# For complexity, since we expect inverted U-shaped relationships, we also run for each feature, a model with only the linear term and a model with also the quadratic term, 
# compare them, and then decide whether we need to include the quadratic term in addition to the linear
# If we need to retain some quadratic terms, then we cannot simply compare the correlations as these assume linear relation only
# Therefore we need to put all the features in a model and select only those that are significant.
# Once we have done this, we then fit a LMM with only the best features for each feature subtype and  report the total R squared for this model

# Concreteness
# correlations with human_int (unrounded)
all_conc_int<-all[,c("PCCNC_megahr","PCCNC_mrc","Int_avg")]
corr_matrix_plot_conc_int<-ggpairs(all_conc_int,upper = list(continuous = wrap("cor", method = "kendall")))
#ggplotly(corr_matrix_plot_conc_int)

#correlations with human_expint (unrounded)
all_conc_expint<-all[,c("PCCNC_megahr","PCCNC_mrc","ExpInt_avg")]
corr_matrix_plot_conc_expint<-ggpairs(all_conc_expint,upper = list(continuous = wrap("cor", method = "kendall")))
#ggplotly(corr_matrix_plot_conc_expint)

# On the basis of both int and exp int, the highest correlation (Kendall) is with MRC-based concreteness

#Complexity

# Decide for each complexity measure whether we should assume a linear or quadratic relation
all_comp<-all[,c("gis","syllable_count","lexicon_count","difficult_words","flesch_reading_ease","flesch_kincaid_grade","smog_index","coleman_liau_index","automated_readability_index","dale_chall_readability_score","spache_readability","gunning_fog","linsear_write_formula","mcalpine_eflaw","text_standard")]
all_comp_int<-all[,c("Int_avg","gis","syllable_count","lexicon_count","difficult_words","flesch_reading_ease","flesch_kincaid_grade","smog_index","coleman_liau_index","automated_readability_index","dale_chall_readability_score","spache_readability","gunning_fog","linsear_write_formula","mcalpine_eflaw","text_standard")]
all_comp_expint<-all[,c("ExpInt_avg","gis","syllable_count","lexicon_count","difficult_words","flesch_reading_ease","flesch_kincaid_grade","smog_index","coleman_liau_index","automated_readability_index","dale_chall_readability_score","spache_readability","gunning_fog","linsear_write_formula","mcalpine_eflaw","text_standard")]

# Use this formula to build polynomial predictor variables

#' Build polynomial predictor variables
#' 
#' Takes a data frame, name of predictor variable, and polynomial order. Creates polynomial-transformed predictor variables, adds them to the data frame and returns the result. Original data frame is unchanged, remember to assign the result.
#' 
#' @param df data frame, should not contain any variables called "predictor"
#' @param predictor string name of predictor variable
#' @param poly.order integer order of polynomial to be created
#' @param orthogonal logical value indicating whether polynomial should be orthogonal (default) or natural (aka raw)
#' @param draw.poly logical value indicating whether to create a graph showing tranformed polynomial predictor values, defaults to TRUE
#' @return Returns a data frame containing the original data and at least two new columns: "predictor".Index, and a column for each order of the polynomial-transformed predictor
#' @examples
#' WordLearnEx.gca <- code.poly(df=WordLearnEx, predictor="Block", poly.order=2)
#' Az.gca <- code.poly(df=Az, predictor="Time", poly.order=2, orthogonal=FALSE)
#' @section Contributors:
#' Originally written by Matt Winn \url{http://www.mattwinn.com/tools/R_add_polynomials_to_df.html} and revised by Dan Mirman to handle various corner cases.
code.poly <- function(df=NULL, predictor=NULL, poly.order=NULL, orthogonal=TRUE, draw.poly=TRUE){
  require(reshape2)
  require(ggplot2)
  # Codes raw or orthogonal polynomial transformations of a predictor variable
  # be sure to not have an actual variable named "predictor" in your data.frame
  # ultimately adds 2 or more columns, including:
  # (predictor).Index, and a column for each order of the polynomials 
  #
  # Written by Matt Winn: http://www.mattwinn.com/tools/R_add_polynomials_to_df.html
  # Revised by Dan Mirman (11/3/2014): 
  #   - call to poly now uses predictor.index and 1:max instead of unique() to deal with out-of-order time bins
  #   - combined indexing and alignment with original data, mainly to avoid problems when poly.order==1
  # Revised by Matt Winn (2/12/2015)
  #   - uses `[` instead of `$` to use variable name to extract column
  #   - computes polynomial on unique(sort(predictor.vector))
  #       rather than 1:max(predictor.vector) 
  #       to accomodate non-integer predictor (e.g. time) levels
  #   - Accomodates missing/unevenly-spaced time bins 
  #       by indexing each sorted unique time bin and using the index to extract
  #       the polynomial value
  
  #===========================================================#
  # convert choice for orthogonal into choice for raw
  raw <- (orthogonal-1)^2
  
  # make sure that the declared predictor is actually present in the data.frame
  if (!predictor %in% names(df)){
    warning(paste0(predictor, " is not a variable in your data frame. Check spelling and try again"))
  }
  
  # Extract the vector to be used as the predictor
  predictor.vector <- df[,which(colnames(df)==predictor)]
  
  # create index of predictor (e.g. numbered time bins)
  # the index of the time bin will be used later as an index to call the time sample
  predictor.indices <- as.numeric(as.factor(predictor.vector))
  
  df$temp.predictor.index <- predictor.indices
  
  #create x-order order polys (orthogonal if not raw)
  predictor.polynomial <- poly(x = unique(sort(predictor.vector)), 
                               degree = poly.order, raw=raw)
  
  # use predictor index as index to align 
  # polynomial-transformed predictor values with original dataset
  # (as many as called for by the polynomial order)
  df[, paste("poly", 1:poly.order, sep="")] <- 
    predictor.polynomial[predictor.indices, 1:poly.order]
  
  # draw a plot of the polynomial transformations, if desired
  if (draw.poly == TRUE){
    # extract the polynomials from the df
    df.poly <- unique(df[c(predictor, paste("poly", 1:poly.order, sep=""))])
    
    # melt from wide to long format
    df.poly.melt <- melt(df.poly, id.vars=predictor)
    
    # Make level names intuitive
    # don't bother with anything above 6th order. 
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly1"] <- "Linear"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly2"] <- "Quadratic"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly3"] <- "Cubic"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly4"] <- "Quartic"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly5"] <- "Quintic"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly6"] <- "Sextic"
    
    # change some column names for the output
    colnames(df.poly.melt)[colnames(df.poly.melt) == "variable"] <- "Order"
    
    poly.plot <- ggplot(df.poly.melt, aes(y=value, color=Order))+
      aes_string(x=predictor)+
      geom_line()+
      xlab(paste0(predictor, " (transformed polynomials)"))+
      ylab("Transformed value")+
      scale_color_brewer(palette="Set1")+
      theme_bw()
    
    print(poly.plot)
  }
  
  # restore correct column names
  colnames(df)[colnames(df) == "temp.predictor.index"] <- paste0(predictor,".Index")
  return(df)
}


models_linear_int = list()
models_quadratic_int = list()
models_linear_expint = list()
models_quadratic_expint = list()
p_int = list()
p_expint = list()
variables = names (all_comp)
for (var in variables){
  #linear int
  models_linear_int[[var]] = lm(
    as.formula(paste0("Int_avg ~ ",var)),
    data=all_comp_int
  )
  #quadratic int
  all_comp_int.poly<-code.poly(df=all_comp_int, predictor=var, poly.order=2, orthogonal=TRUE, draw.poly=FALSE)
  models_quadratic_int[[var]] = lm(
    as.formula("Int_avg ~ poly1 + poly2"),
    data=all_comp_int.poly
  )
  
  anova_var_int<-anova(models_quadratic_int[[var]],models_linear_int[[var]])
  p_int[[var]] = anova_var_int$'Pr(>F)'[2]
  
  #linear expint
  models_linear_expint[[var]] = lm(
    as.formula(paste0("ExpInt_avg ~ ",var)),
    data=all_comp_expint
  )
  #quadratic expint
  all_comp_expint.poly<-code.poly(df=all_comp_expint, predictor=var, poly.order=2, orthogonal=TRUE, draw.poly=FALSE)
  models_quadratic_expint[[var]] = lm(
    as.formula("ExpInt_avg ~ poly1 + poly2"),
    data=all_comp_expint.poly
  )
  
  anova_var_expint<-anova(models_quadratic_expint[[var]],models_linear_expint[[var]])
  p_expint[[var]] = anova_var_expint$'Pr(>F)'[2]
}

complexity_var<-data.frame(Var=variables, A_Int = NA, A_ExpInt = NA)
for (var in variables){
  complexity_var$A_Int[complexity_var$Var==var]<-p_int[[var]]
  complexity_var$A_ExpInt[complexity_var$Var==var]<-p_expint[[var]]
}
complexity_var$S_Int<-ifelse(complexity_var$A_Int<.05,"quadratic","linear")
complexity_var$S_ExpInt<-ifelse(complexity_var$A_ExpInt<.05,"quadratic","linear")
#complexity_var


### NEW GISPY
# Var        A_Int     A_ExpInt     S_Int  S_ExpInt
# 1                           gis 6.982122e-06 2.683757e-03 quadratic quadratic
# 2                syllable_count 7.742194e-84 2.608049e-47 quadratic quadratic
# 3                 lexicon_count 1.551893e-78 1.762496e-44 quadratic quadratic
# 4               difficult_words 3.456976e-49 2.233137e-27 quadratic quadratic
# 5           flesch_reading_ease 8.620321e-07 1.029247e-05 quadratic quadratic
# 6          flesch_kincaid_grade 2.406889e-04 1.368175e-03 quadratic quadratic
# 7                    smog_index 2.274761e-01 3.825074e-01    linear    linear
# 8            coleman_liau_index 1.675675e-25 3.251546e-17 quadratic quadratic
# 9   automated_readability_index 5.579212e-01 5.556150e-01    linear    linear
# 10 dale_chall_readability_score 4.341986e-14 5.799036e-07 quadratic quadratic
# 11           spache_readability 1.531104e-01 3.249087e-01    linear    linear
# 12                  gunning_fog 4.385448e-20 7.199777e-13 quadratic quadratic
# 13        linsear_write_formula 6.369871e-15 1.244060e-09 quadratic quadratic
# 14               mcalpine_eflaw 3.767093e-16 3.050512e-10 quadratic quadratic
# 15                text_standard 9.005492e-10 1.472971e-06 quadratic quadratic

# This suggests all complexity measures should be entered as quadratic predictors except smog_index, automated_readability_index and spache_readability

#Complexity model
#INT
complexity_model_int<-lmer(Int_avg~poly(gis,2)+poly(syllable_count,2) +poly(lexicon_count,2)+poly(difficult_words,2) +
                             poly(flesch_reading_ease,2)+poly(flesch_kincaid_grade,2)+poly(coleman_liau_index,2)+
                             poly(dale_chall_readability_score,2)+poly(gunning_fog,2)+poly(linsear_write_formula,2)+
                             poly(mcalpine_eflaw,2)+poly(text_standard,2)+
                             smog_index + automated_readability_index + spache_readability+
                             (1|conversation_id)+(1|project), data = all)

### NEW GISPY
# Random effects:
#   Groups          Name        Variance Std.Dev.
# conversation_id (Intercept) 0.05188  0.2278  
# project         (Intercept) 0.15364  0.3920  
# Residual                    0.28680  0.5355  
# Number of obs: 6656, groups:  conversation_id, 64; project, 32
# 
# Fixed effects:
#   Estimate Std. Error t value
# (Intercept)                              2.607472   0.155521  16.766
# poly(gis, 2)1                            1.053922   0.713187   1.478
# poly(gis, 2)2                           -1.742993   0.573352  -3.040
# poly(syllable_count, 2)1               -12.257696  10.473360  -1.170
# poly(syllable_count, 2)2                 6.540259   5.177131   1.263
# poly(lexicon_count, 2)1                 24.719307   9.419836   2.624
# poly(lexicon_count, 2)2                -13.722945   4.724873  -2.904
# poly(difficult_words, 2)1                4.790877   2.257288   2.122
# poly(difficult_words, 2)2               -2.814434   1.270433  -2.215
# poly(flesch_reading_ease, 2)1            4.536368   8.298289   0.547
# poly(flesch_reading_ease, 2)2           -1.628939   1.345551  -1.211
# poly(flesch_kincaid_grade, 2)1          13.363152  13.710639   0.975
# poly(flesch_kincaid_grade, 2)2           5.034945   3.405455   1.478
# poly(coleman_liau_index, 2)1             9.127261   1.610936   5.666
# poly(coleman_liau_index, 2)2            -0.203655   0.783245  -0.260
# poly(dale_chall_readability_score, 2)1  -1.629206   0.863509  -1.887
# poly(dale_chall_readability_score, 2)2   0.132342   0.621688   0.213
# poly(gunning_fog, 2)1                    5.356327   2.503579   2.139
# poly(gunning_fog, 2)2                   -5.438981   2.285789  -2.379
# poly(linsear_write_formula, 2)1         -4.210631   4.482570  -0.939
# poly(linsear_write_formula, 2)2          5.595377   2.806209   1.994
# poly(mcalpine_eflaw, 2)1                 4.577858   6.396852   0.716
# poly(mcalpine_eflaw, 2)2                -3.335750   2.265676  -1.472
# poly(text_standard, 2)1                 -1.134878   0.987069  -1.150
# poly(text_standard, 2)2                  1.442000   0.812286   1.775
# smog_index                               0.007823   0.002775   2.819
# automated_readability_index             -0.019323   0.006964  -2.774
# spache_readability                      -0.101386   0.030948  -3.276

#The following complexity predictors are significant
# gis (quadratic, neg)
# lexicon count (linear, pos + quadratic, neg)
# difficult_words (linear, pos + quadratic, neg)
# coleman liau index (linear, pos)
# gunning_fog (linear, pos + quadratic, neg)
# smog_index (linear, pos)
# automated readability_index (linear, neg)
# spache_readability (linear, neg)

#To generate a latex table
#kable(data.frame(coef(summary(complexity_model_int))), format ="latex", caption=paste(format(formula(complexity_model_int)),collapse = ''), col.names=c("Feature","B","SE","t"), digits=3)

#EXP_INT
complexity_model_expint<-lmer(ExpInt_avg~poly(gis,2)+poly(syllable_count,2) +poly(lexicon_count,2)+poly(difficult_words,2) +
                                poly(flesch_reading_ease,2)+poly(flesch_kincaid_grade,2)+poly(coleman_liau_index,2)+
                                poly(dale_chall_readability_score,2)+poly(gunning_fog,2)+poly(linsear_write_formula,2)+
                                poly(mcalpine_eflaw,2)+poly(text_standard,2)+
                                smog_index + automated_readability_index + spache_readability+
                                (1|conversation_id)+(1|project), data = all)
###NEW GISPY
# Random effects:
#   Groups          Name        Variance Std.Dev.
# conversation_id (Intercept) 0.04051  0.2013  
# project         (Intercept) 0.13155  0.3627  
# Residual                    0.27220  0.5217  
# Number of obs: 6656, groups:  conversation_id, 64; project, 32
# 
# Fixed effects:
#   Estimate Std. Error t value
# (Intercept)                              2.495849   0.149545  16.690
# poly(gis, 2)1                            0.292432   0.694512   0.421
# poly(gis, 2)2                           -0.733678   0.558474  -1.314
# poly(syllable_count, 2)1               -14.180129  10.201746  -1.390
# poly(syllable_count, 2)2                 8.940835   5.043243   1.773
# poly(lexicon_count, 2)1                 26.760713   9.175622   2.917
# poly(lexicon_count, 2)2                -14.473916   4.602706  -3.145
# poly(difficult_words, 2)1                1.856175   2.198691   0.844
# poly(difficult_words, 2)2               -1.858529   1.237521  -1.502
# poly(flesch_reading_ease, 2)1           11.477167   8.083215   1.420
# poly(flesch_reading_ease, 2)2           -1.613695   1.310709  -1.231
# poly(flesch_kincaid_grade, 2)1          25.388617  13.355307   1.901
# poly(flesch_kincaid_grade, 2)2           1.312530   3.317225   0.396
# poly(coleman_liau_index, 2)1             8.653992   1.568994   5.516
# poly(coleman_liau_index, 2)2             0.117073   0.762976   0.153
# poly(dale_chall_readability_score, 2)1  -0.790785   0.840884  -0.940
# poly(dale_chall_readability_score, 2)2   0.868966   0.605606   1.435
# poly(gunning_fog, 2)1                    3.179775   2.438761   1.304
# poly(gunning_fog, 2)2                   -4.097099   2.226654  -1.840
# poly(linsear_write_formula, 2)1         -5.615281   4.366534  -1.286
# poly(linsear_write_formula, 2)2          6.845372   2.733609   2.504
# poly(mcalpine_eflaw, 2)1                -0.819599   6.231025  -0.132
# poly(mcalpine_eflaw, 2)2                -1.942566   2.207127  -0.880
# poly(text_standard, 2)1                  0.472163   0.961556   0.491
# poly(text_standard, 2)2                  1.616143   0.791297   2.042
# smog_index                               0.006318   0.002703   2.337
# automated_readability_index             -0.023435   0.006783  -3.455
# spache_readability                      -0.090391   0.030144  -2.999

#The following complexity predictors are significant
# lexicon count (linear, pos)
# lexicon count (quadratic, neg)
# coleman_liau_index (linear, pos)
# linsear_write_formula (quadratic, pos)
# text_standard (quadratic, pos)
# smog_index (linear, pos)
# automated_readability_index (linear, neg)
# spache_readability (linear, neg)

# Comparison between INT and EXP INT
### NEW GISPY
#INT
#The following complexity predictors are significant
# gis (quadratic, neg)
# lexicon count (linear, pos + quadratic, neg)
# difficult_words (linear, pos + quadratic, neg)
# coleman liau index (linear, pos)
# gunning_fog (linear, pos + quadratic, neg)
# smog_index (linear, pos)
# automated readability_index (linear, neg)
# spache_readability (linear, neg)

#EXP_INT
#The following complexity predictors are significant
# lexicon count (linear, pos + quadratic, neg)
# coleman_liau_index (linear, pos)
# linsear_write_formula (quadratic, pos)
# text_standard (quadratic, pos)
# smog_index (linear, pos)
# automated_readability_index (linear, neg)
# spache_readability (linear, neg)

#Significant predictors shared between INT and EXP_INT
# lexicon count (linear, pos)
# lexicon count (quadratic, neg)
# coleman_liau_index (linear, pos)
# smog_index (linear, pos)
# automated readability_index (linear, neg)
# spache_readability (linear, neg)

#To generate a latex table
#kable(data.frame(coef(summary(complexity_model_expint))), format ="latex", caption=paste(format(formula(complexity_model_expint)),collapse = ''), col.names=c("Feature","B","SE","t"), digits=3)


# How do we know how much variance is explained by the model?
# http://bbolker.github.io/mixedmodels-misc/glmmFAQ.html#how-do-i-compute-a-coefficient-of-determination-r2-or-an-analogue-for-glmms
# https://psycnet.apa.org/doiLanding?doi=10.1037%2Fmet0000184 (r2MLM)

## Uptake (5)
# 4 uptake measures are in this file
uptakes<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/all_uptakes_final.csv",header=T)
#select unique doccano_id's
uptakes<-uptakes %>% distinct(id, conversation_id, .keep_all=T)
#summary(uptakes)
#str(uptakes)
#the relevant measures are:
# cos_proc_within_page (note: replace 0 with NA as these are cases where there was only one speaker, and then replace all other measure with NA too)
# LCS 
# perTinS
# perSinT
names(uptakes)[1]<-"doccano_id"
names(uptakes)[21]<-"LCS_proc"
uptakes<-uptakes[,-9] #remove human_int (this is the non-averaged value from one annotator)
#Check in how many cases cos is 0
summary(uptakes$cos_proc_within_page==0)#150
summary(uptakes$cos_within_page==0)#64
summary(uptakes$cos_pages==0)#128
summary(uptakes$cos_proc_pages==0)#129

#replace cases where cos is == 0 with NA (across all uptake measures)
uptakes[uptakes$cos_within_page==0|uptakes$cos_proc_within_page==0,-c(1:2)]<-NA

#merge
all<-merge(all,uptakes, by=c("doccano_id","conversation_id"))

#an additional uptake measure (model-based, from the educonvokit) is in this file
uptakes_eck<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/convokit_uptake.csv",header=T)
names(uptakes_eck)[1]<-"doccano_id"
all<-merge(all,uptakes_eck, by=c("doccano_id"))

#This measure is based on a model of teacher uptake of student turns, so we only look at teacher_uptake_student
#For pages where the first speaker is the student, then the teacher and student turn this refers to are on the same page
#But for pages where the first speaker is the teacher, then the measure is the teacher uptake of the student on the previous page
#We also have student_teacher_uptake that may be more meaningful on teacher-first pages (as the student turn was being rated), 
#but we need to be careful with interpretation as the model was specifically trained on teacher uptake of the student, not vice versa.
#Note there are many missing values (presumably because the measure can not always be computed if there was no previous teacher/student turn)

#Check the distributions of all uptake measures (use _proc version except for cosine similarity measures where we consider both)
# These measures have very skewed distributions, so they will be dichotomised.
#summary(all$LCS_proc)
#ggplot(all,aes(x=LCS_proc))+geom_density()
all$LCS_proc_d<-ifelse(all$LCS_proc>0,"high","low")
#summary(as.factor(all$LCS_proc_d))
# high  low NA's 
# 2131 4375  150 

#summary(all$perTinS)
#ggplot(all,aes(x=perTinS))+geom_density()
all$perTinS_d<-ifelse(all$perTinS>0,"high","low")
#summary(as.factor(all$perTinS_d))
# high  low NA's 
# 1932 4574  150 

#summary(all$perSinT)
#ggplot(all,aes(x=perSinT))+geom_density()
all$perSinT_d<-ifelse(all$perSinT>0,"high","low")
#summary(as.factor(all$perSinT_d))
# high  low NA's 
# 1932 4574  150 

#perTinS and perSinT, when dichtomised, are identical (see below, there ar 150 NAs), so only use one of them
# sum(all$perSinT_d == all$perTinS_d,na.rm=T)

#summary(all$teacher_uptake_student)
#ggplot(all,aes(x=teacher_uptake_student))+geom_density()
all$tushl<-ifelse(all$teacher_uptake_student>0.95,"high","low")
#summary(as.factor(all$tushl))
# high  low NA's 
# 1805 3749 1102 

# ggplot(all,aes(y=Int_avg,x=tushl,col=tushl,linetype=first_role))+geom_boxplot()+
#   stat_summary(aes(y=Int_avg,x=tushl,shape=first_role),fun="mean", geom="point", size=3, position=position_dodge())

#summary(all$student_uptake_teacher)
#ggplot(all,aes(x=student_uptake_teacher))+geom_density()
all$suthl<-ifelse(all$student_uptake_teacher>0.95,"high","low")
#summary(as.factor(all$suthl))
# high  low NA's 
# 1694 4622  340 

#ggplot(all,aes(y=Int_avg,x=suthl,col=suthl,linetype=first_role))+geom_boxplot()+
#  stat_summary(aes(y=Int_avg,x=suthl,shape=first_role),fun="mean", geom="point", size=3, position=position_dodge())

# These measures are roughly normally distributed so do not need dichotomising
# summary(all$cos_proc_within_page)
# ggplot(all,aes(x=cos_proc_within_page))+geom_density()
# 
# summary(all$cos_within_page)
# ggplot(all,aes(x=cos_within_page))+geom_density()

## Selecting uptake variables for combined model
all$LCS_proc_d_num<-ifelse(all$LCS_proc_d=="high",.5,-.5)
all$LCS_proc_d_numc<-scale(all$LCS_proc_d_num,T,F)

all$perTinS_d_num<-ifelse(all$perTinS_d=="high",.5,-.5)
all$perTinS_d_numc<-scale(all$perTinS_d_num,T,F)

all$frc<-ifelse(all$first_role=="student"&is.na(all$first_role)==F,.5,ifelse(all$first_role=="teacher"&is.na(all$first_role)==F,-.5,NA))
all$frcc<-scale(all$frc,T,F)
all$tushl<-ifelse(all$teacher_uptake_student>0.95,.5,-.5)
all$suthl<-ifelse(all$student_uptake_teacher>0.95,.5,-.5)
all$tushlc<-scale(all$tushl,T,F)
all$suthlc<-scale(all$suthl,T,F)

all$cos_proc_within_page_c<-scale(all$cos_proc_within_page,T,F)
all$cos_within_page_c<-scale(all$cos_within_page,T,F)

#Note the following models ony include 5275 observations because of NAs (particularly with the edoconvokit measure)
m_uptake_int<-lmer(Int_avg~1+LCS_proc_d_numc+perTinS_d_numc+ frcc*tushlc+frcc*suthlc+cos_proc_within_page_c+cos_within_page_c+(1|project)+(1|conversation_id),data=all)
#summary(m_uptake_int)

#To generate a latex table
#kable(data.frame(coef(summary(m_uptake_int))), format ="latex", caption=paste(format(formula(m_uptake_int)),collapse = ''), col.names=c("Feature","B","SE","t"), digits=3)

m_uptake_expint<-lmer(ExpInt_avg~1+LCS_proc_d_numc+perTinS_d_numc+ frcc*tushlc+frcc*suthlc+cos_proc_within_page_c+cos_within_page_c+(1|project)+(1|conversation_id),data=all)
#summary(m_uptake_expint)

#To generate a latex table
#kable(data.frame(coef(summary(m_uptake_expint))), format ="latex", caption=paste(format(formula(m_uptake_expint)),collapse = ''), col.names=c("Feature","B","SE","t"), digits=3)

# All measures have a significant relation to Int - this is however negative for cos similarity (higher similarity = less interest)
# All measures except propTinS have a significant relation to ExpInt, but the relation is again negative, not positive, for cos similarity.

## Combined models
## These will include the following predictors
## Concreteness: PCCNC_mrc
## Comprehensibility: lexicon count (linear + quadratic), coleman-liau-index (linear), smog-index (linear), automated readability-index (linear), spache_readability (linear), GIS (linear + quadratic)
## Uptake: LCS_proc, student-uptake-teacher, cos-within-pages (raw)

## We ran models with random intercepts for annotator and conversation, and random slopes by annotator for all features (if it converges)
# First we need to reshape the dataset from wide to long format

names(all)[124:126]<-c("AnnId_1", "AnnId_2", "AnnId_3")
all.long<-reshape(all,varying = names(all)[c(117,118,120,121,122,123,124,125,126)],sep="_",direction="long")

# scale all variables
all.long$conc<-scale(all.long$PCCNC_mrc,T,T)
all.long$cli<-scale(all.long$coleman_liau_index,T,T)
all.long$si<-scale(all.long$smog_index,T,T)
all.long$ari<-scale(all.long$automated_readability_index,T,T)
all.long$sri<-scale(all.long$spache_readability,T,T)
all.long$LCS_proc_d_num<-ifelse(all.long$LCS_proc_d=="high",.5,-.5)
all.long$LCS_proc_d_numc<-scale(all.long$LCS_proc_d_num,T,T)

all.long$suthl<-ifelse(all.long$student_uptake_teacher>0.95,.5,-.5)
all.long$suthlc<-scale(all.long$suthl,T,T)

all.long$cos_within_page_c<-scale(all.long$cos_within_page,T,T)

all.long$gis_lc<-scale(poly(all.long$gis,2)[,1],T,T)
all.long$gis_qc<-scale(poly(all.long$gis,2)[,2],T,T)
all.long$lex_lc<-scale(poly(all.long$lexicon_count,2)[,1],T,T)
all.long$lex_qc<-scale(poly(all.long$lexicon_count,2)[,2],T,T)


#did not converge
#combined_int<-lmer(Int~conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c+ (1|conversation_id) +(1+ conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c||AnnId), data=all.long)

#remove random slopes
combined_int<-lmer(Int~conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c+ (1|conversation_id) +(1|AnnId), data=all.long)
#summary(combined_int)

#To generate a latex table
#kable(data.frame(coef(summary(combined_int))), format ="latex", caption=paste(format(formula(combined_int)),collapse = ''), col.names=c("Feature","Beta","SE","t"), digits=3)

combined_expint<-lmer(ExpInt~conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c+ (1|conversation_id) +(1|AnnId), data=all.long)
#summary(combined_expint)

#To generate a latex table
#kable(data.frame(coef(summary(combined_expint))), format ="latex", caption=paste(format(formula(combined_expint)),collapse = ''), col.names=c("Feature","Beta","SE","t"), digits=3)

#variability explained by AnnId is about 3 times as large as variability explained by conversation

#Generate figure that illustrate relations between key features and int/expint
#We want three rows, each with two columns (int and exp int)
coni<-ggplot(all.long,aes(x=conc,y=Int))+geom_boxplot(aes(x=conc,y=as.factor(Int)),alpha=0.3)+geom_smooth(method="lm")+xlab("MEGAHR Concreteness (scaled)") + ylab("Interestingness") + theme_cowplot() + theme(legend.position="none")
cone<-ggplot(all.long,aes(x=conc,y=ExpInt))+geom_boxplot(aes(x=conc,y=as.factor(Int)),alpha=0.3)+geom_smooth(method="lm")+xlab("MEGAHR Concreteness (scaled)") + ylab("Expected interestingness") + theme_cowplot() + theme(legend.position="bottom")

#split lexicon count into 3 equal subsets
all.long$lex_equal<-cut_number(all.long$lexicon_count, n = 3, labels = c("Low", "Medium", "High"))
all.long$gis_equal<-cut_number(all.long$gis_lc, n = 3, labels = c("Low", "Medium", "High"))

gisi<-ggplot(all.long,aes(x=gis_lc,y=Int, col=lex_equal))+geom_jitter(alpha=0.4, height=0.4, width=0.8)+geom_smooth(method="loess", col="black")+xlab("GIS score (scaled)") + ylab("Interestingness") + theme_cowplot()+ scale_color_grey(name="Lexicon count") + theme(legend.position="none")
gise<-ggplot(all.long,aes(x=gis_lc,y=ExpInt, col=lex_equal))+geom_jitter(alpha=0.4, height=0.4, width=0.8)+geom_smooth(method="loess", col="black")+xlab("GIS score (scaled)") + ylab("Expected interestingness") + theme_cowplot()+ scale_color_grey(name="Lexicon count") + theme(legend.position="bottom")

all.long$uptake_factor<-as.factor(all.long$suthlc)
levels(all.long$uptake_factor)<-c("low","high")
all.long$LCS_factor<-as.factor(all.long$LCS_proc_d_numc)
levels(all.long$LCS_factor)<-c("low","high")
upi<-ggplot(all.long,aes(x=uptake_factor,y=Int, col=LCS_factor))+geom_boxplot()+stat_summary(aes(y=Int,x=uptake_factor, col=LCS_factor),fun="mean", geom="point", size=3)+xlab("Student uptake teacher") + ylab("Interestingness") + theme_cowplot() + scale_color_grey(name="LCS") + theme(legend.position="none")
upe<-ggplot(all.long,aes(x=uptake_factor,y=ExpInt, col=LCS_factor))+geom_boxplot()+stat_summary(aes(y=ExpInt,x=uptake_factor, col=LCS_factor),fun="mean", geom="point", size=3)+xlab("Student uptake teacher") + ylab("Expected interestingness") + theme_cowplot() + scale_color_grey(name="LCS") + theme(legend.position="bottom")

plot_grid(coni, cone, gisi, gise, upi, upe, align = "h", ncol=2)
ggsave("features.png", width=22, height=24,units = "cm" )
# concreteness with slopes
combined_int_conc<-lmer(Int~conc+ (1|conversation_id) +(1+conc||AnnId), data=all.long)
#summary(combined_int_conc)

combined_expint_conc<-lmer(ExpInt~conc+ (1|conversation_id) +(1+conc||AnnId), data=all.long)
#summary(combined_expint_conc)

#comprehensibility with slopes

#combined_int_comp<-lmer(Int~cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+ (1|conversation_id) +(1+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc||AnnId), data=all.long)
#did not converge
# remove ari and sri (ns in combined model)
#combined_int_comp<-lmer(Int~cli+si+gis_lc+gis_qc+lex_lc+lex_qc+ (1|conversation_id) +(1+cli+si+gis_lc+gis_qc+lex_lc+lex_qc||AnnId), data=all.long)
#did not converge
#remove smallest slope (si)
#combined_int_comp<-lmer(Int~cli+si+gis_lc+gis_qc+lex_lc+lex_qc+ (1|conversation_id) +(1+cli+gis_lc+gis_qc+lex_lc+lex_qc||AnnId), data=all.long)
#did not converge
#also remove next smallest slope (cli)
#combined_int_comp<-lmer(Int~cli+si+gis_lc+gis_qc+lex_lc+lex_qc+ (1|conversation_id) +(1+gis_lc+gis_qc+lex_lc+lex_qc||AnnId), data=all.long)
#did not converge
#also remove nect smallest slope (gis_qc)
combined_int_comp<-lmer(Int~cli+si+gis_lc+gis_qc+lex_lc+lex_qc+ (1|conversation_id) +(1+gis_lc+lex_lc+lex_qc||AnnId), data=all.long)
#converged
#summary(combined_int_comp)

#use the same model for expint
combined_expint_comp<-lmer(ExpInt~cli+si+gis_lc+gis_qc+lex_lc+lex_qc+ (1|conversation_id) +(1+gis_lc+lex_lc+lex_qc||AnnId), data=all.long)
#summary(combined_expint_comp)

#uptake with slopes
combined_int_up<-lmer(Int~LCS_proc_d_numc+suthlc+cos_within_page_c+ (1|conversation_id) +(1+LCS_proc_d_numc+suthlc+cos_within_page_c||AnnId), data=all.long)
#summary(combined_int_up)

combined_expint_up<-lmer(ExpInt~LCS_proc_d_numc+suthlc+cos_within_page_c+ (1|conversation_id) +(1+LCS_proc_d_numc+suthlc+cos_within_page_c||AnnId), data=all.long)
#summary(combined_expint_up)

#Compute explained variance usinf r2MLM
#https://psycnet.apa.org/doiLanding?doi=10.1037%2Fmet0000184 (r2MLM)


# we need to drop one random factor (drop conversation_id because it explains less variance)
#convert all variables to numeric
all.long$Int<-as.numeric(all.long$Int)
all.long$ExpInt<-as.numeric(all.long$ExpInt)
all.long$conc<-as.numeric(all.long$conc)
all.long$cli<-as.numeric(all.long$cli)
all.long$si<-as.numeric(all.long$si)
all.long$ari<-as.numeric(all.long$ari)
all.long$sri<-as.numeric(all.long$sri)
all.long$LCS_proc_d_numc<-as.numeric(all.long$LCS_proc_d_numc)
all.long$suthlc<-as.numeric(all.long$suthlc)

all.long$cos_within_page_c<-as.numeric(all.long$cos_within_page_c)

all.long$gis_lc<-as.numeric(all.long$gis_lc)
all.long$gis_qc<-as.numeric(all.long$gis_qc)
all.long$lex_lc<-as.numeric(all.long$lex_lc)
all.long$lex_qc<-as.numeric(all.long$lex_qc)

combined_int_of<-lmer(Int~conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c +(1|AnnId), data=all.long)
summary(combined_int_of)
r2mlm(combined_int_of,bargraph=T)

#do the same for exp_int final model
combined_expint_of<-lmer(ExpInt~conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c +(1|AnnId), data=all.long)
summary(combined_expint_of)
r2mlm(combined_expint_of,bargraph=T)

# and now with models per feature group
# concreteness
combined_int_conc_of<-lmer(Int~conc +(1+conc||AnnId), data=all.long)
r2mlm(combined_int_conc_of,bargraph=T)

combined_expint_conc_of<-lmer(ExpInt~conc +(1+conc||AnnId), data=all.long)
r2mlm(combined_expint_conc_of,bargraph=T)

#comprehensibility
combined_int_comp_of<-lmer(Int~cli+si+gis_lc+gis_qc+lex_lc+lex_qc +(1+gis_lc+gis_qc+lex_lc+lex_qc||AnnId), data=all.long)
r2mlm(combined_int_comp_of,bargraph=T)

#combined_expint_comp_of<-lmer(ExpInt~cli+si+gis_lc+gis_qc+lex_lc+lex_qc +(1+gis_lc+gis_qc+lex_lc+lex_qc||AnnId), data=all.long)
# did not converge
# remove gis_lc because it has the smallest estimate
combined_expint_comp_of<-lmer(ExpInt~cli+si+gis_qc+lex_lc+lex_qc +(1+gis_qc+lex_lc+lex_qc||AnnId), data=all.long)
summary(combined_expint_comp_of)
r2mlm(combined_expint_comp_of,bargraph=T)

#uptake
combined_int_up_of<-lmer(Int~LCS_proc_d_numc+suthlc+cos_within_page_c+(1+LCS_proc_d_numc+suthlc+cos_within_page_c||AnnId), data=all.long)
r2mlm(combined_int_up_of,bargraph=T)
combined_expint_up_of<-lmer(ExpInt~LCS_proc_d_numc+suthlc+cos_within_page_c+(1+LCS_proc_d_numc+suthlc+cos_within_page_c||AnnId), data=all.long)
r2mlm(combined_expint_up_of,bargraph=T)
# note that no within cluster variance is graphed here despite the inclusion of random slopes.

#add student and annotator level variables to the combined models (without random slopes)
####### IMPORTANT #####
#### These analyses need to be rerun with correct annotator info from

# First, we need to get the annotator proficiency from a separate file
prof<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/proficiency.csv")
names(prof)<-c("AnnId","annotator_level")

#remove incorrect annotator_level column from all.long
all.long<-subset(all.long, select= -c(annotator_level))

#strip final _int from AnnId in all.long
all.long$AnnId<-gsub("_int","",all.long$AnnId)

#add missing AnnId to prof and set annotator_level to NA

for (i in unique(all.long$AnnId)){
  if (!(i %in% prof$AnnId)){
      new_id<-data.frame(AnnId = i, annotator_level= NA)
      prof<-rbind(prof,new_id)
    }
}

#now merge
all.long<-merge(all.long,prof,by=c("AnnId"))

# turn them into numeric predictors and centre
all.long$student_level_nc<-as.numeric(scale(as.numeric(as.factor(all.long$student.cefr.level)),T,F))
all.long$annotator_level_nc<-as.numeric(scale(as.numeric(as.factor(all.long$annotator_level)),T,F))
all.long$level_match<-ifelse(all.long$student.cefr.level==all.long$annotator_level,"yes","no")
all.long$level_match_num<-ifelse(all.long$level_match=="yes",.5,-.5)
all.long$level_match_numc<-as.numeric(scale(all.long$level_match_num,T,F))

combined_int_as<-lmer(Int~level_match_numc+student_level_nc+ annotator_level_nc+ conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c+(1|AnnId), data=all.long)
summary(combined_int_as)
r2mlm(combined_int_as,bargraph=T)
combined_expint_as<-lmer(ExpInt~level_match_numc+student_level_nc+ annotator_level_nc+conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c +(1|AnnId), data=all.long)
summary(combined_expint_as)
r2mlm(combined_expint_as,bargraph=T)

# summary(as.factor(all.long$level_match))
# summaryBy(Int~level_match+student.cefr.level, FUN=mean, data=all.long)
# summaryBy(Int~student.cefr.level, FUN=mean, data=all.long)
# summaryBy(Int~annotator_level, FUN=mean, data=all.long)

#Figure
all.long %>%
  group_by(level_match,student.cefr.level) %>%
  mutate(ssize = n()) %>%
  ggplot(aes(x=level_match, y=Int, col=student.cefr.level))+ 
  stat_summary(aes(size=ssize),fun="mean", geom="pointrange", fun.max = function(x) mean(x) + sd(x),
               fun.min = function(x) mean(x) - sd(x), position=position_dodge(width=0.5))+
  guides(size = guide_legend(override.aes = list(size = 1), title=""),col=guide_legend(title="Student CEFR level")) + ylab("Interestingness") + xlab("Student-Annotator Level Match")+
  theme_bw() + theme(axis.text=element_text(size=15),legend.text = element_text(size=15),axis.title = element_text(size=20), legend.title = element_text(size=15))
ggsave("proficiency.png", width=10,height=6)

#Linguistic features and variance in interest/expected interest ratings
# Concreteness
ggplot(all,aes(x=PCCNC_megahr,y=Int_var)) + geom_jitter() + geom_smooth(method = "loess")

# Comprehensibility
# lexicon count (quadratic), coleman-liau-index (linear), smog-index (linear), automated readability-index (linear), GIS (quadratic), spache_readability (linear)

ggplot(all,aes(x=lexicon_count,y=Int_var)) + geom_jitter() + geom_smooth(method = "loess")
ggplot(all,aes(x=gis,y=Int_var)) + geom_jitter() + geom_smooth(method = "loess")
ggplot(all,aes(x=coleman_liau_index,y=Int_var)) + geom_jitter() + geom_smooth(method = "loess")
ggplot(all,aes(x=smog_index,y=Int_var)) + geom_jitter() + geom_smooth(method = "loess")
ggplot(all,aes(x=automated_readability_index,y=Int_var)) + geom_jitter() + geom_smooth(method = "loess")
ggplot(all,aes(x=spache_readability,y=Int_var)) + geom_jitter() + geom_smooth(method = "loess")

# Uptake: LCS_proc, student-uptake-teacher, cos-within-pages (raw)

ggplot(all,aes(x=LCS_proc_d,y=Int_var)) + geom_boxplot()
ggplot(all,aes(x=as.factor(suthl),y=Int_var)) + geom_boxplot()
ggplot(all,aes(x=cos_within_page,y=Int_var)) + geom_jitter() + geom_smooth(method = "loess")


#Overall, no indication that Int_var depends on any of the features
# scale all variables
all$conc<-scale(all$PCCNC_mrc,T,T)
all$cli<-scale(all$coleman_liau_index,T,T)
all$si<-scale(all$smog_index,T,T)
all$ari<-scale(all$automated_readability_index,T,T)
all$sri<-scale(all$spache_readability,T,T)
all$LCS_proc_d_num<-ifelse(all$LCS_proc_d=="high",.5,-.5)
all$LCS_proc_d_numc<-scale(all$LCS_proc_d_num,T,T)

all$suthl<-ifelse(all$student_uptake_teacher>0.95,.5,-.5)
all$suthlc<-scale(all$suthl,T,T)

all$cos_within_page_c<-scale(all$cos_within_page,T,T)

all$gis_lc<-scale(poly(all$gis,2)[,1],T,T)
all$gis_qc<-scale(poly(all$gis,2)[,2],T,T)
all$lex_lc<-scale(poly(all$lexicon_count,2)[,1],T,T)
all$lex_qc<-scale(poly(all$lexicon_count,2)[,2],T,T)

combined_int_var<-lmer(Int_var~conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c+ (1|conversation_id) +(1|project), data=all)
summary(combined_int_var)
#only lex_qc (positive) and LCS_proc_d_numc (negative)
#summaryBy(Int_var~LCS_proc_d_num,FUN=mean, data=all)

combined_expint_var<-lmer(ExpInt_var~conc+cli+si+ari+sri+gis_lc+gis_qc+lex_lc+lex_qc+LCS_proc_d_numc+suthlc+cos_within_page_c+ (1|conversation_id) +(1|project), data=all)
summary(combined_expint_var)
#none

### Human Reward Prediction error

all_rpe<- all %>%
  group_by(conversation_id) %>%
  arrange(.,doccano_id,by_group=T)%>%
  mutate(human_rpe = Int_avg-dplyr::lag(ExpInt_avg,order_by=doccano_id))%>%
  slice(.,-1)%>%
  ungroup()

all_rpe<-as.data.frame(all_rpe)

# In addition, we should also use cos similarity between pages
# There are still some zero values in these measures
# 145 NA are cases where the within-page value was 0, which were excluded from previous analyses
#summary(all_rpe$cos_proc_pages)
#summary(all_rpe$cos_pages)

#code the remaining zero values as NA
all_rpe$cos_proc_pages[all_rpe$cos_proc_pages==0]<-NA #6 more
all_rpe$cos_pages[all_rpe$cos_pages==0]<-NA #5 more

#scale
all_rpe$cos_proc_pages_c<-scale(all_rpe$cos_proc_pages,T,T)
all_rpe$cos_pages_c<-scale(all_rpe$cos_pages,T,T)

#Sequence type (note only the first turn of a sequence is labelled)
# Extract topic opening
#summary(as.factor(all_rpe$seq.type))
all_rpe$sequence<-ifelse(str_detect(all_rpe$seq.type,"topic opening"),"opening","not opening")
#summary(as.factor(all_rpe$sequence))

all_rpe$opening<-ifelse(all_rpe$sequence=="opening",0.5,-0.5)
all_rpe$opening_scale<-scale(all_rpe$opening, T, T)

#summaryBy(human_rpe+Int_avg+ExpInt_avg~as.factor(opening), FUN=mean, data=all_rpe)
# opening human_rpe.mean Int_avg.mean ExpInt_avg.mean
# 1    -0.5     0.09970333     2.100180        1.991100
# 2     0.5     0.43870968     2.133333        1.982796
# 3      NA     0.13103448     1.528736        1.645977


#rpe predicted by sequence
sequence_rpe_model<-lmer(human_rpe~1+opening_scale + (1|conversation_id)+(1|project), data=all_rpe)
summary(sequence_rpe_model)

#latex table
kable(data.frame(coef(summary(sequence_rpe_model))), format ="latex", caption=paste(format(formula(sequence_rpe_model)),collapse = ''), col.names=c("Feature","Beta","SE","t"), digits=3)


#cos_pages predicted by sequence
#sequence_cos_pages_model<-lmer(cos_pages_c~1+opening_scale + (1|conversation_id)+(1|project), data=all_rpe)
#singular
# remove random intercept by project, estimated at zero
#summary(sequence_cos_pages_model)
sequence_cos_pages_model<-lmer(cos_pages_c~1+opening_scale + (1|conversation_id), data=all_rpe)
summary(sequence_cos_pages_model)# ns

#summaryBy(cos_pages~as.factor(opening),FUN=mean, data=all_rpe, na.rm=T)
# opening cos_pages.mean
# 1    -0.5      0.7488523
# 2     0.5      0.7407720
# 3      NA            NaN

#sequence_cos_pages_model_proc<-lmer(cos_proc_pages_c~1+opening_scale + (1|conversation_id) + (1|project), data=all_rpe)
#singular
# remove random intercept by project, estimated at zero
sequence_cos_pages_model_proc<-lmer(cos_proc_pages_c~1+opening_scale + (1|conversation_id), data=all_rpe)
summary(sequence_cos_pages_model_proc)#negative

#summaryBy(cos_proc_pages~as.factor(opening),FUN=mean, data=all_rpe, na.rm=T)
# opening cos_proc_pages.mean
# 1    -0.5           0.5802272
# 2     0.5           0.5523795
# 3      NA                 NaN

cos_rpe_model<-lmer(human_rpe~1+cos_pages_c + (1|conversation_id)+(1|project), data=all_rpe)
summary(cos_rpe_model)
# processed
cos_rpe_model_proc<-lmer(human_rpe~1+cos_proc_pages_c+ (1|conversation_id)+(1|project), data=all_rpe)
summary(cos_rpe_model_proc)

#only topic opening
cos_rpe_model_proc_to<-lmer(human_rpe~1+cos_proc_pages_c+ (1|conversation_id)+(1|project), data=subset(all_rpe, sequence=="opening"))
summary(cos_rpe_model_proc_to) #ns

#ggplot(all_rpe,aes(human_rpe,cos_proc_pages)) +geom_jitter() + geom_smooth(method="loess")

#get rid of missing values
all_rpe<-all_rpe[is.na(all_rpe$cos_proc_pages)==F,]

#linear and quadratic effect
all_rpe$cpp_lc<-scale(poly(all_rpe$cos_proc_pages,2)[,1],T,T)
all_rpe$cpp_qc<-scale(poly(all_rpe$cos_proc_pages,2)[,2],T,T)
cos_rpe_model_proc_q<-lmer(human_rpe~1+cpp_lc+cpp_qc+ (1|conversation_id)+(1|project), data=all_rpe)
summary(cos_rpe_model_proc_q)

##########################################
### Interest-ratings generated by LLMs ###
##########################################

#This script will merge all trial_level data from the INTEREST project (i.e., all human annotations at the level of individual turns)
#with all LLM generated interest ratings from the same turns
#and all turn-level or turn-pair-level linguistic features

#The features and LLM ratings (and human average INT ratings) are in this file
all_feat_m<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/all_llms_human.csv")
#this has 6656 rows
#str(all_feat)
#conversation_id
#id (Doccano-generated, should be larger for later projects and later pages within a project)

#add gpt4o with history
gpt4<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/all_gpt4_ratings.csv",header=T)
gpt4$gpt4_int<-as.character(gpt4$gpt4_int)
gpt4$gpt4_int[is.na(gpt4$gpt4_int)]<-"None"
summary(as.factor(gpt4$gpt4_int))
all_feat_m<-merge(all_feat_m,gpt4,by=c("id","conversation_id","gpt4_int"))
#add llama3
llama3<-read.csv("https://raw.githubusercontent.com/mahathi-p/Interest-in-humans-and-llms/refs/heads/main/data/all_llama_ratings.csv",header=T)
summary(llama3)
####NOTE####
# the latest version for 13b llama is correct (all_llama_ratings), do NOT use the version in all_llms_human.csv
# remove X13b_int from all_feat
all_feat_m<-all_feat_m[,-110]
all_feat_m<-merge(all_feat_m,llama3,by=c("id","conversation_id","X7b_int","X7bnonchat_int","X13bnonchat_int"))

#Make sure all missing values are coded as NA
# first drop all irrelevant columns that contain strings
all_feat_m<-select(all_feat_m,-c("text","proc_text.x","proc_text.y"))
all_feat_m[all_feat_m=="None"]<-NA

all_feat_m<-as.data.frame(imap_dfr(all_feat_m, ~ as.numeric(.x)))
  
#generate table with missing and > 4 scores for all models
oldnames<-c("X7b_int", "X7bnonchat_int", "X13bnonchat_int", "gpt4_int", "new_13bchat", "new_7bchat", "mis_nonchat_int", "mix_int", "mis7_int", "gpt4o_int", "X13b_int", "llama3_chat_int", "llama3_nonchat_int")
newnames<-paste(oldnames,".Missing",sep="")
newnames2<-paste(oldnames,".Impossible",sep="")
table.wide<-all_feat_m%>%
  select(X7b_int, X7bnonchat_int, X13bnonchat_int, gpt4_int, new_13bchat, new_7bchat, mis_nonchat_int, mix_int, mis7_int, gpt4o_int, X13b_int, llama3_chat_int, llama3_nonchat_int)%>%
  summarise(across(everything(), ~ sum(is.na(.))))%>%
  rename_with(~newnames[which(oldnames ==.x)], .cols=all_of(oldnames))%>%
  cbind (all_feat_m%>%
           select(X7b_int, X7bnonchat_int, X13bnonchat_int, gpt4_int, new_13bchat, new_7bchat, mis_nonchat_int, mix_int, mis7_int, gpt4o_int, X13b_int, llama3_chat_int, llama3_nonchat_int)%>%
           summarise(across(everything(), ~ sum(.>4,na.rm = T)))%>%
           rename_with(~newnames2[which(oldnames ==.x)], .cols=all_of(oldnames))
)

table.long<-reshape(table.wide,varying=names(table.wide),direction="long")
table.long<-table.long[,-c(1,15)]

#reorder columns
table.long<-table.long[,c(2,3,1,11,6,5,13,12,7,9,8,4,10)]

# column name key:
#   LLAMA:
#   "X7b_int"   - llama2 7B chat     (TheBloke/Llama-2-7B-chat-GPTQ)
# "X13b_int"   - llama2 13B chat    (TheBloke/Llama-2-13B-chat-GPTQ)
# "X7bnonchat_int"     - llama2 7B non chat   (used Xingwei's prompt coz mine didnt work for nonchat)      
# "X13bnonchat_int"    - llama2 13B non chat  (used Xingwei's prompt)     
# "new_13bchat"   - llama2 13B chat  (chat with the non-chat prompt by Xingwei)   
# "new_7bchat"     - llama2 7B chat  (chat with the non-chat prompt by Xingwei)   
# 
# "llama3_chat_int"   - llama3 8B  chat     (similar to what I used for llama2 chat)
# "llama3_nonchat_int"  - llama 3 8B non chat with xingwei's prompt to maintain consistency. all the llama2,3 nonchat responses are with the same prompt.
# 
# MISTRAL: 
#   "mis_nonchat_int"   - mistral 7b non chat (mistralai/Mistral-7B-v0.1)
# "mix_int"    -       mixtral  7x8b instruct (mistralai/Mixtral-8x7B-Instruct-v0.1)
# "mis7_int" -   mistral 7b instruct (mistralai/Mistral-7B-Instruct-v0.2) 
# 
# GPT4:
#   "gpt4_int" - gpt 4              
# "gpt4o_int"   - gpt4o    
#

# Set the vector of names and save it to a variable 
llms<-c("llama2 7B non-chat", "llama2 13B non-chat", "llama2 7B chat (IB prompt)", "llama2 13B chat (IB prompt)","llama2 7B chat (CB prompt)","llama2 13B chat (CB prompt)","llama3 8B non-chat","llama3 8B chat","mistral 7B non-chat","mistral 7B instruct","mixtral 8X7B instruct","gpt4","gpt4o")

#transpose
table.long$Row.Names<-rownames(table.long)
table.long<-table.long[,c(14,1:13)]
names(table.long)<-c("Row.Names",llms)
table.long2 <- data.frame(t(table.long[-1]))
colnames(table.long2) <- table.long[,1]

#generate a latex table now
kable(table.long2, format ="latex", caption="Count of missing and impossible ratings by LLM",col.names = c("Missing (count)","Values above 4 (count)"), digits=3)

#Fix issues with scores > 4
all_feat_m$X7b_int[all_feat_m$X7b_int>4]<-(4)
all_feat_m$X13b_int[all_feat_m$X13b_int>4]<-(4)
all_feat_m$mis_nonchat_int[all_feat_m$mis_nonchat_int>4]<-(4)
all_feat_m$mis7_int[all_feat_m$mis7_int>4]<-(4)

# # covert all to numeric
# all_feat$X7b_int<-as.numeric(all_feat$X7b_int)
# all_feat$X13b_int<-as.numeric(all_feat$X13b_int)
# all_feat$X7bnonchat_int<-as.numeric(all_feat$X7bnonchat_int)
# all_feat$X13bnonchat_int<-as.numeric(all_feat$X13bnonchat_int)
# all_feat$new_7bchat<-as.numeric(all_feat$new_7bchat)
# all_feat$new_13bchat<-as.numeric(all_feat$new_13bchat)
# all_feat$gpt4_int<-as.numeric(all_feat$gpt4_int)
# all_feat$mis_nonchat_int<-as.numeric(all_feat$mis_nonchat_int)
# all_feat$mis7_int<-as.numeric(all_feat$mis7_int)
# all_feat$mix_int<-as.numeric(all_feat$mix_int)
# all_feat$gpt4o_int<-as.numeric(all_feat$gpt4o_int)
# all_feat$llama3_chat_int<-as.numeric(all_feat$llama3_chat_int)
# all_feat$llama3_nonchat_int<-as.numeric(all_feat$llama3_nonchat_int)

#merge with trial_level data (all_trial)

names(all_feat_m)[1]<-"doccano_id"

allm<-merge(all_feat_m,all_trial,by=c("conversation_id","doccano_id"))
#compute human_expint
allm$human_expint<-round(allm$ExpInt_avg)

#### Correlation plot with human_int and all model ratings (use human_int because it is rounded to an integer and the models gave integer scores)
allm<-allm[,c(1,2,105,134,4,5,3,114,109,108,116,115,110,112,111,6,113)]
names(allm)[5:17]<-llms

ggc<-ggcorr(allm[,-c(1:2)],method=c("pairwise","kendall"), label=T,layout.exp=3, label_round=2, label_alpha=F, midpoint=0.1, size=0, label_size = 8)
# Create labels and plot
dat <- data.frame(x = seq(allm[,-c(1:2)]), y = seq(allm[,-c(1:2)]), 
                  lbs = gsub("[0:9]B ","[0:9]B\n",gsub("_", " ", names(allm[,-c(1:2)])) ))
models_corr<-ggc + geom_text(data=dat, aes(x, y, label=lbs), nudge_x = 2, hjust=0.7, size=10) + theme(legend.text = element_text(size=20))
ggsave("models_crr.png",plot=models_corr, width=20, height = 15)

#compute cohen's kappa
#chat models
kappa2(allm[,c("human_int","gpt4o")]) #0.0994
kappa2(allm[,c("human_int","gpt4")]) #0.032
kappa2(allm[,c("human_int","mixtral 8X7B instruct")]) #0.0732
kappa2(allm[,c("human_int","llama3 8B chat")]) #0.0456

#non-chat models 
kappa2(allm[,c("human_int","llama2 7B non-chat")]) #0.0153
kappa2(allm[,c("human_int","llama2 13B non-chat")]) #0.0126
kappa2(allm[,c("human_int","mistral 7B non-chat")]) #-0.00104
kappa2(allm[,c("human_int","llama3 8B non-chat")]) #0.00573

#### Correlate model error (human-model) with variance in human int score (Int_var)
all_error<-allm[,-c(1:2,4)]
all_error[2:ncol(all_error)]<-all_error[,1]-all_error[2:ncol(all_error)]
all_error<-cbind(all_error,all[,1:2])

#merge with all_trial to get Int_var
all_error<-merge(all_error,all_trial,by=c("doccano_id","conversation_id"))

#Chat models average and variance
#Choose the four chat models that have highest Kendall tau correlations with human_int
# Chose 4 rather than 3 because I wanted to include the best performing llama3 model (which ranked fourth)

#add Int_avg from all_trial
allm<-merge(allm,all_trial,by=c("doccano_id","conversation_id"))

allm<-allm%>%
  rowwise()%>%
  mutate(
    chat_model_avg = mean(c(gpt4o,gpt4,`mixtral 8X7B instruct`,`llama3 8B chat`),na.rm=T,trim=0),
    chat_model_var = var(c(gpt4o,gpt4,`mixtral 8X7B instruct`,`llama3 8B chat`),na.rm=T),
    chat_model_error = Int_avg - chat_model_avg,
    nonchat_model_avg = mean(c(`llama2 7B non-chat`,`llama2 13B non-chat`,`mistral 7B non-chat`,`llama3 8B non-chat`),na.rm=T,trim=0),
    nonchat_model_var = var(c(`llama2 7B non-chat`,`llama2 13B non-chat`,`mistral 7B non-chat`,`llama3 8B non-chat`),na.rm=T),
    nonchat_model_error = Int_avg - nonchat_model_avg
  )

summary(allm$chat_model_var)
summary(allm$nonchat_model_var)

summary(abs(allm$chat_model_error))
summary(abs(allm$nonchat_model_error))

cnc<-allm[,c("chat_model_var","nonchat_model_var","chat_model_error","nonchat_model_error","conversation_id","doccano_id")]
names(cnc)<-c("var.chat","var.nonchat","error.chat","error.nonchat","conversation_id","doccano_id")
cnc.long<-reshape(cnc, varying=names(cnc)[1:4],idvar=c("conversation_id","doccano_id"),direction="long", timevar="Type")

ggplot(cnc.long,aes(x=Type,y=var, col=Type))+geom_pirate() + stat_summary(aes(x=Type,y=abs(error)), fun.y=mean, geom="point", size=5, col="black") + stat_summary(aes(x=Type,y=abs(error)), fun.data=mean_cl_boot, geom="errorbar", width=0.3, col="black") + 
  theme_bw()+ylab ("Variance/Absolute Error") + scale_color_brewer(type="qual") +
  theme(axis.title = element_text(size=20),axis.text = element_text(size=15))

ggsave("model_error.png", height=6,width=4)
