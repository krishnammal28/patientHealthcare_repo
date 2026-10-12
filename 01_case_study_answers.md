# Case Study: Predicting Patient Re-admission in Healthcare
Course: 231CSV65 – Predictive Data Science

## a) Problem faced by the hospital (2 Marks)
Johns Hopkins Hospital faces a high rate of **30-day patient readmissions**, i.e. patients being admitted again within 30 days of discharge. This raises treatment costs, overloads beds and staff, lowers care quality scores, and can attract financial penalties. The hospital needs to identify *which patients are likely to be readmitted* before they leave, so preventive action can be taken.

## b) Input data used for building the predictive model (3 Marks)
The model is built on **Electronic Health Record (EHR)** data:
1. **Patient medical history** – age, gender, chronic conditions (diabetes, heart disease, COPD), past admissions, medications.
2. **Laboratory results** – blood glucose, haemoglobin, creatinine, blood pressure, and other test values.
3. **Hospital stay details** – length of stay, diagnosis, procedures performed, ICU stay, discharge type and discharge destination.

The target variable is binary: **Readmitted within 30 days (1) / Not readmitted (0)**. Data is cleaned (missing values, outliers), encoded and scaled before training.

## c) How predictive analytics identifies high-risk patients (4 Marks)
1. **Data collection** – historical EHR records of discharged patients with known readmission outcomes are gathered.
2. **Preprocessing & feature selection** – missing values handled, categorical features encoded, important predictors (age, number of prior admissions, comorbidities, abnormal labs, length of stay) selected.
3. **Model training** – a classification algorithm (Logistic Regression, Decision Tree, Random Forest, etc.) learns patterns that separate readmitted from non-readmitted patients.
4. **Validation** – the model is tested on unseen data using accuracy, precision, recall and ROC-AUC.
5. **Risk scoring** – for each new patient the model outputs a **probability (risk score)** of readmission; patients above a threshold are flagged as **high risk**.

## d) Actions taken and benefits (5 Marks)
**Actions taken for high-risk patients**
- Detailed discharge planning and patient education.
- Follow-up appointments or calls within 48–72 hours of discharge.
- Medication reconciliation and adherence support.
- Home-care visits or tele-monitoring.
- Referral to specialists and care coordination with primary care physicians.

**Benefits**
- Reduced 30-day readmission rate.
- Lower hospital costs and avoidance of regulatory penalties.
- Better use of beds, staff and resources (targeted, not blanket, interventions).
- Improved patient outcomes and satisfaction.
- Data-driven, proactive instead of reactive care.

## e) Interpretation of the chart and role of predictive analytics (5 Marks)
> **Note:** The chart referred to in this question is not visible in the question paper you provided. The interpretation below is for the standard chart of this case – readmission rate / number of high-risk patients **before vs. after** using the predictive model. Please adjust the numbers to match your actual chart.

**Interpretation**
- Before the model, the readmission rate is high and interventions are applied uniformly or not at all.
- After deployment, the chart shows the readmission rate **falling** over time, since patients flagged as high-risk receive follow-up.
- The gap between the two lines/bars represents the **readmissions prevented** by the model.

**Role of predictive analytics**
- Provides **early warning** before discharge.
- Enables **targeted** use of limited resources on the patients who need them most.
- Supports continuous monitoring: the model is retrained as new data arrives.
- Converts raw EHR data into actionable insight, reducing readmissions and cost.

## f) How predicted risk scores help healthcare providers make decisions (5 Marks)
A risk score (e.g. 0–100% probability) lets providers **prioritise and personalise** care:
- **Low risk (e.g. < 30%)** – standard discharge instructions and routine follow-up.
- **Medium risk (30–60%)** – phone follow-up, medication review, patient education.
- **High risk (> 60%)** – extended stay if needed, home visits, specialist referral, close monitoring, care-manager assignment.

Further uses:
- **Resource allocation** – care managers and beds are assigned where risk is highest.
- **Discharge timing** – a high score may delay discharge until the patient is stabilised.
- **Explainability** – the top contributing factors (e.g. multiple prior admissions, abnormal lab values) tell clinicians *what* to address.
- **Threshold tuning** – hospitals balance sensitivity (catching more at-risk patients) against workload and cost.
- **Performance tracking** – scores can be compared with actual outcomes to improve the model.

Risk scores **support, not replace,** clinical judgement.
