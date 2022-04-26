About the project and its goal:

The project was made in Polish language because it concerns Polish medical unit.

Creation of an application that allows the decision-maker, to carry out the process of classification of the sum of costs of hospitalization of a new patient
of the hospital, who is at the stage of admission to the institution, even before the implementation of the appropriate method of treatment.
The list of the most important functions that the tool is equipped with is listed below:

- Loading and automatic processing of the medical facility patient database, 
- Selection of a department to which a new patient is to be admitted,
- Selection of the number of hospitalisation cost ranges to be created for a selected department,
- Selection of the classification model with which the prediction is to be made,
- Teaching the model on training data and then checking the classification performance metrics of the trained model on test data,
- Performing a prediction of the cost of a patient's hospitalization, using the trained model,
  based on the parameters for the patient's admission to the selected department,
- Displaying the result of the prediction performed.

About data:

Data obtained by the author, come from hospital situated in Poland. These are aggregate billing data on hospitalization costs for patients of a selected hospital.

Variables explanation:

- ID_PACJENT - which is a unique identifier assigned to each patient, 
- ROZP_GLOWNE- containing the code assigned to the disease for which the patient is being treated, 
- TRYB_PRZYJECIA_KOD - which contains the code for the patient's mode of admission to hospital, 
- TYP_KOMORKI_ORG - contains code of the unit where the patient was admitted, 
- MC_SPRAWOZDAWCZY - contains information about the year of the event, 
- ROK_SPRAWOZDAWCZY - contains information about the month of the event, 
- WARTOSC_SKOR - contains the adjusted benefit cost value for each row of the table.

![image](https://user-images.githubusercontent.com/71133618/164339392-38721d42-db8d-44fa-ae30-5a081773ce4d.png)

![image](https://user-images.githubusercontent.com/71133618/164339480-d997b7b0-a3fe-4eb7-a699-c3caa3ef2bd9.png)

![image](https://user-images.githubusercontent.com/71133618/164339591-4905f277-8948-47ee-a869-1f4b5a914265.png)
