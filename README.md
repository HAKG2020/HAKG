# HAKG

## Introduction 

Hirerarchical Attentive Knowledge Graph Embedding (HAGE) is a new recommendation framework tailored to knowledge graph-based personalized recommendation. HKGE exploits the user-item connectivities in knowledge graphs with expressive subgraphs to provide better recommendation.

## Environment Requirement
+ Python 2.7
+ Pytorch (cudatoolkit=9.0)
+ numpy == 1.16.2
+ networkx == 2.2

## Datasets

+ MovieLens
   + For the MoiveLens dataset, we crawl the corresponding IMDB dataset as movie auxiliary information, including genre, director, and actor. Note that we automatically remove the movies without auxilairy information. We then combined MovieLens and IMDB by movie title and released year. The combined data is saved in a txt file (movie-features-unmapping.txt) and the format is as follows:    
   
         id:1|actors:Tom Hanks, Tim Allen, Don Rickles, Jim Varney|director:John Lasseter|genre:Animation, Adventure, Comedy
   
   + For the original user-movie rating data, we remove all items without auxiliary information. The data is save in a txt file (user_movies.txt) and the format is as follows:  
   
         userid itemid rating timestamp
   
+ Last-FM

   + This is the music listening dataset collected from Last.fm online music systems. Wherein, the tracks are viewed as the items. In particular, we take the subset of the dataset where the timestamp is from Jan, 2015 to June, 2015. For Last-FM,we map items into Freebase entities via title matching if there is a mapping available. The overall KG is saved in kg_final.txt and the format is as follows:

         head_entity_id  relation_id  tail_entity_id
   
+ Yelp
   + It records user ratings on local business scaled from 1-5. Additionally, social relations as well as business attributes (e.g., category, city) are also included. For Yelp, we extract item knowledge from the local business information network (e.g., category, location,
and attribute) as KG data. The format is as follows:

         id:11163|genre:Accountants,Professional Services,Tax Services,Financial Services|city:Peoria
      
## Modules 

For clarify, hereafter we use movieLens dataset as a toy example to demonstrate the detailed modules of HKGE. 

+ Data Split (split_train_test.py)

   + Split the user-movie rating data into training and test data

   + Input Data: user_movies.txt

   + Output Data: training.txt, test.txt

+ Negative Sample (negative_sample_for_train.py, negative_sample_for_test.py)

   + Sample negative movies for each user to balance the model training & Sample negative movies for test. 
    
   + Input Data: training.txt; test.txt
   
   + Output Data: negative.txt; test_negative.txt

+ Path Sampling （path_positive.py, path_negative.py, path_test_negative.py）

   + Extract paths for positive and negative user-moive interaction and prepare paths data for test, respectively.
   
   + Input Data: user-movie interaction for positive/negative/test rating
   
   + Output Data: sampled path for positive/negtiave/test user-item pair

+ Train and Test (Train.py, Main.py)

   + Feed both postive and negative path into the HAKG, train and evaluate the model
   
   + You can run Main.py to start training and testing the model. 
 
   
## References

   [1] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. Learningentity and relation embeddings for knowledge graph      completion. In AAAI.

   [2] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Ok-sana Yakhnenko. 2013.  Translating embeddings for        modeling multi-relationaldata. In NIPS. 2787–2795.

   [3] Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. Knowledgegraph embedding by translating on hyperplanes. In AAAI.
