# TML1M

## Overview

**Table-Movielens1M (TML1M)** is a relational table dataset extends the original Movielens-1M dataset [1][2] with enriched movie data. It includes three tables: users, movies and ratings. The "users" and the "movies" contain several key attributes related to users and movies. The "ratings" captures the relationships between specific users and movies. This new version contains 6,040 users, 3,883 movies, and 1,000,209 ratings. 

## Data Processing

The processing focused exclusively on the movies, extracting "time-invariant" metadata from each film. This metadata includes details such as the director, cast, running time, language, certification, plot, and URL, obtained by using the Movielens ID to access the corresponding movie page on movielens.org [3]. Afterward, the movie IDs were reordered to be consecutive for improved usability.


In addition to these adjustments, several discrepancies were identified:
- Movie with Movielens ID 2228 (MovieID 2160) is missing from Movielens, and data from IMDB was used as a substitute.
- Movie with Movielens IDs 1741 (MovieID 1691) and 1758 (MovieID 1706) are duplicates of 1795 (MovieID 1736) and 2563 (MovieID 2495), respectively.

## Dataset Composition

- **users.csv**: This file contains 6040 users, each user has UserID, Gender, Age, Occupation and Zip-code. Occupations are shown in numerical code form. Age is chosen from the following ranges:
	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"

- **movies.csv**: This file includes 3883 movies with MovieID, Title, Year, Genre, Director, Cast, Runtime, Languages, Certificate, Plot and Url information.
- 
    * MovieID: The unique identifier for each movie, organized in sequential order.
    * Title: The title of the movie, sourced from the Movielens website.
    * Year: The release year of the movie, obtained from the Movielens website.
    * Genre: The genres of the movie, derived from the original Movielens-1M dataset, separated by vertical bars (“|”).
    * Director: The name(s) of the director(s) of the movie, sourced from the Movielens website, listed in commas if there is more than one.
    * Cast: The primary cast of the movie, obtained from the Movielens website, separated by commas.
    * Runtime: The duration of the movie, sourced from the Movielens website.
    * Languages: The official language versions of the movie, obtained from the Movielens website, separated by commas if there are multiple.
    * Certificate: The movie certification information, sourced from the Movielens website.
    * Plot: A brief summary of the movie's main plot, sourced from the Movielens website.
    * URL: The URL of the movie on the Movielens website.


- **ratings.csv**: This file contains 1,000,209 ratings, each row of data represents a specific user's evaluation of a particular movie, composed of UserID, MovieID, Rating and Timestamp.
- **<span style="color: black;">masks.pt</span>**: This file divides the dataset into training (140), verification (500), and test (1,000) sets. The training set includes 20 users from each age group, while the verification and test sets are structured to align as closely as possible with the natural distribution of users.
- **embeddings.npy**: This file contains embeddings for movie information generated using the “all-MiniLM-L6-v2” model.

## References
[1]. GroupLens. (2015). MovieLens 1M dataset. Retrieved from https://grouplens.org/datasets/movielens/1m/

[2]. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), Article 19. https://doi.org/10.1145/2827872

[3]. GroupLens. (2013). MovieLens. from https://movielens.org/

## Citing

If you find this useful in your research, please cite our paper, thx:
```
@article{rllm2024,
      title={rLLM: Relational Table Learning with LLMs}, 
      author={Weichen Li and Xiaotong Huang and Jianwu Zheng and Zheng Wang and Chaokun Wang and Li Pan and Jianhua Li},
      year={2024},
      eprint={2407.20157},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.20157}, 
}
```


---

### ---------- The Following is Movielens-1M's Original README ----------

SUMMARY
================================================================================

These files contain 1,000,209 anonymous ratings of approximately 3,900 movies 
made by 6,040 MovieLens users who joined MovieLens in 2000.

USAGE LICENSE
================================================================================

Neither the University of Minnesota nor any of the researchers
involved can guarantee the correctness of the data, its suitability
for any particular purpose, or the validity of results based on the
use of the data set.  The data set may be used for any research
purposes under the following conditions:

     * The user may not state or imply any endorsement from the
       University of Minnesota or the GroupLens Research Group.

     * The user must acknowledge the use of the data set in
       publications resulting from the use of the data set
       (see below for citation information).

     * The user may not redistribute the data without separate
       permission.

     * The user may not use this information for any commercial or
       revenue-bearing purposes without first obtaining permission
       from a faculty member of the GroupLens Research Project at the
       University of Minnesota.

If you have any further questions or comments, please contact GroupLens
<grouplens-info@cs.umn.edu>. 

CITATION
================================================================================

To acknowledge use of the dataset in publications, please cite the following
paper:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872


ACKNOWLEDGEMENTS
================================================================================

Thanks to Shyong Lam and Jon Herlocker for cleaning up and generating the data
set.

FURTHER INFORMATION ABOUT THE GROUPLENS RESEARCH PROJECT
================================================================================

The GroupLens Research Project is a research group in the Department of 
Computer Science and Engineering at the University of Minnesota. Members of 
the GroupLens Research Project are involved in many research projects related 
to the fields of information filtering, collaborative filtering, and 
recommender systems. The project is lead by professors John Riedl and Joseph 
Konstan. The project began to explore automated collaborative filtering in 
1992, but is most well known for its world wide trial of an automated 
collaborative filtering system for Usenet news in 1996. Since then the project 
has expanded its scope to research overall information filtering solutions, 
integrating in content-based methods as well as improving current collaborative 
filtering technology.

Further information on the GroupLens Research project, including research 
publications, can be found at the following web site:
        
        http://www.grouplens.org/

GroupLens Research currently operates a movie recommender based on 
collaborative filtering:

        http://www.movielens.org/

RATINGS FILE DESCRIPTION
================================================================================

All ratings are contained in the file "ratings.dat" and are in the
following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040 
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings

USERS FILE DESCRIPTION
================================================================================

User information is in the file "users.dat" and is in the following
format:

UserID::Gender::Age::Occupation::Zip-code

All demographic information is provided voluntarily by the users and is
not checked for accuracy.  Only users who have provided some demographic
information are included in this data set.

- Gender is denoted by a "M" for male and "F" for female
- Age is chosen from the following ranges:

	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"

- Occupation is chosen from the following choices:

	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"

MOVIES FILE DESCRIPTION
================================================================================

Movie information is in the file "movies.dat" and is in the following
format:

MovieID::Title::Genres

- Titles are identical to titles provided by the IMDB (including
year of release)
- Genres are pipe-separated and are selected from the following genres:

	* Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western

- Some MovieIDs do not correspond to a movie due to accidental duplicate
entries and/or test entries
- Movies are mostly entered by hand, so errors and inconsistencies may exist
