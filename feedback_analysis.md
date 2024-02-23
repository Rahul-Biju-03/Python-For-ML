# Report on Analysis of Feedback Data for Course Improvement

## 1) Problem Statement:
The aim of this project is to analyze feedback data collected from participants in various courses or training sessions to identify patterns, trends, and areas for improvement. The feedback data consists of ratings on different aspects such as content quality, training effectiveness, expertise of resource persons, and session organization, along with additional comments and suggestions provided by the participants.

## 2) Introduction:
In an educational institution or training program, gathering feedback from participants is crucial for assessing the effectiveness of the courses offered and understanding areas that require improvement. The feedback provides valuable insights into the quality of content, training methods, and overall organization of the sessions. Analyzing this feedback data can aid in making informed decisions to enhance the learning experience and satisfaction of the participants.

## 3) Dataset Details:
The dataset contains 174 rows (entries) and 12 columns (fields).
Contains the following columns
1) Timestamp ,2) Name of the Participant ,3) Email ID ,4) Branch ,5) Semester ,6) Recourse Person of the session ,7) Rating for the overall quality and relevance of the course content presented in this session ,8) Rating for the extent of effectiveness of the training methods and delivery style in helping understand the concepts presented ,9) Rating for the resource person's knowledge and expertise in the subject matter covered during this session ,10) Rating for the extent of relevance and applicability of the content covered in this session to real-world industry scenarios ,11) Rating for the overall organization of the session, including time management,clarity of instructions,and interactive elements ,12) Additional comments, suggestions, or feedback regarding the session.

## 4) Methodology

### a) Data Collection:
The feedback data was gathered via Google Forms, where participants provided their responses. The dataset includes responses from participants, detailing their perceptions and suggestions regarding course content, effectiveness, and overall organization.

### b) Data Wrangling

The data wrangling process includes dropping irrelevant columns such as "Timestamp","Email ID" and "Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience." from the DataFrame and renaming the remaining columns to more descriptive names. This helps streamline the dataset by removing unnecessary information and organizing the relevant data in a more understandable format for further analysis. The renamed columns are "Name","Branch","Semester","Resourse Person","Content Quality","Effeciveness","Expertise","Relevance","Overall Organization".

### c) Exploratory Data Analysis approach:

#### 1. Frequency Calculation:
Determining the occurrence of each unique value in the dataset, specifically focusing on the "Resource Person" and "Name" columns.

#### 2. Percentage Calculation:
Approach: Converting the frequency counts into percentages to analyze the relative contribution of each resource person and participant.

#### 3. Rounding:
Approach: Rounding the calculated percentages to improve readability and presentation.

#### Justifications:
Relevance: Analyzing the distribution of feedback data across resource persons and participants is relevant as it sheds light on the engagement levels and participation patterns within the course.
Insight Generation: Understanding how feedback data is distributed facilitates the generation of insights into trends, preferences, and potential areas for improvement. It helps in identifying effective resource persons and active participants.
Decision Support: The insights derived from EDA serve as a basis for data-driven decision-making in course administration and improvement efforts. Course administrators can allocate resources more efficiently, tailor course content to meet participants' needs, and address any issues identified through the analysis.

### d) ML approach:


## 5) EDA

### + Percentage analysis of Resource_person wise distribution of data.

Resourse Person
Mrs. Akshara Sasidharan    34.48
Mrs. Veena A Kumar         31.03
Dr. Anju Pratap            17.24
Mrs. Gayathri J L          17.24

### * Percentage analysis of Name wise distribution of data

Name
Sidharth V Menon             4.02
Rizia Sara Prabin            4.02
Aaron James Koshy            3.45
Rahul Krishnan               3.45
Allen John Manoj             3.45
Christo Joseph Sajan         3.45
Jobinjoy Ponnappal           3.45
Varsha S Panicker            3.45
Nandana A                    3.45
Anjana Vinod                 3.45
Rahul Biju                   3.45
Kevin Kizhakekuttu Thomas    3.45
Lara Marium Jacob            3.45
Abia Abraham                 3.45
Shalin Ann Thomas            3.45
Abna Ev                      3.45
Aaron Thomas Blessen         2.87
Sebin Sebastian              2.87
Sani Anna Varghese           2.87
Bhagya Sureshkumar           2.87
Jobin Tom                    2.87
Leya Kurian                  2.87
Jobin Pius                   2.30
Aiswarya Arun                2.30
Muhamed Adil                 2.30
Marianna Martin              2.30
Anaswara Biju                2.30
Mathews Reji                 1.72
MATHEWS REJI                 1.72
Riya Sara Shibu              1.72
Riya Sara Shibu              1.72
Aiswarya Arun                1.15
Sarang kj                    1.15
Muhamed Adil                 1.15
Lisbeth Ajith                1.15
Jobin Tom                    0.57
Lisbeth                      0.57
Anaswara Biju                0.57
Aaron Thomas Blessen         0.57
Lisbeth Ajith                0.57
Marianna Martin              0.57

### c) Visualisation

 * Subplot regarding Faculty-wise distribution of data
<p align="left">
  <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/5abbbda1-2a27-4fce-b56d-111342f551b1" alt="[faculty wise distribution of data]" width="500">
</p>

 * Piechart regarding Resource person distribution of data
<p align="left">
  <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/864c76b4-ccb0-4da2-834a-39036d9ec940" alt="[resource person]" width="500">
</p>


<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/f5494a6e-88df-4e9a-9e65-4bccf38bb588" alt="SUM 1" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/72ef6282-bbef-4c49-bc76-9e905d01ddd4" alt="SUM 2" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/411818a6-1dd4-4ead-b8e3-da7419d33834" alt="SUM 3" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/e3d4cc49-3f79-4490-8c70-8e8cc9776f21" alt="SUM 4" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/76eb6d4b-4182-4994-bac7-b986bd633db9" alt="SUM 5" width="500" style="float:left;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/f18482be-8d69-4f57-ae14-6a9b334e7ea8" alt="SUM 6" width="500" style="float:right;">
</div>
<div style="margin: 0 auto; width: 500px;">
    <img src="https://github.com/Rahul-Biju-03/Python-For-ML/assets/106422354/8f6f116a-912b-404f-977d-9a8f5c2994f1" alt="SUM 7" width="400">
</div>











## Machine Learning Model to study segmentation: K-means clustering



## Results and conclusion

![Image Description](visualisation.jpg)

- Item 1
  - Nested Item 1
  - Nested Item 2
- Item 2
- Item 3

- **Bold Item 1**
- **Bold Item 2**
- **Bold Item 3**




