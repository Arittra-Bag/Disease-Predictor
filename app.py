import gradio as gr
import pandas as pd
import joblib

# Load the model and label encoder
model = joblib.load('disease_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define a function for predictions
def predict_disease(
    itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering,
    chills, joint_pain, stomach_pain, acidity, ulcers_on_tongue,
    muscle_wasting, vomiting, burning_micturition, spotting_urination, fatigue,
    weight_gain, anxiety, cold_hands_and_feets, mood_swings, weight_loss,
    restlessness, lethargy, patches_in_throat, irregular_sugar_level, cough,
    high_fever, sunken_eyes, breathlessness, sweating, dehydration,
    indigestion, headache, yellowish_skin, dark_urine, nausea, loss_of_appetite,
    pain_behind_the_eyes, back_pain, constipation, abdominal_pain, diarrhoea,
    mild_fever, yellow_urine, yellowing_of_eyes, acute_liver_failure, fluid_overload,
    swelling_of_stomach, swelled_lymph_nodes, malaise, blurred_and_distorted_vision,
    phlegm, throat_irritation, redness_of_eyes, sinus_pressure, runny_nose,
    congestion, chest_pain, weakness_in_limbs, fast_heart_rate, pain_during_bowel_movements,
    pain_in_anal_region, bloody_stool, irritation_in_anus, neck_pain, dizziness,
    cramps, bruising, obesity, swollen_legs, swollen_blood_vessels, puffy_face_and_eyes,
    enlarged_thyroid, brittle_nails, swollen_extremeties, excessive_hunger, extra_marital_contacts,
    drying_and_tingling_lips, slurred_speech, knee_pain, hip_joint_pain, muscle_weakness,
    stiff_neck, swelling_joints, movement_stiffness, spinning_movements, loss_of_balance,
    unsteadiness, weakness_of_one_body_side, loss_of_smell, bladder_discomfort,
    foul_smell_of_urine, continuous_feel_of_urine, passage_of_gases, internal_itching,
    toxic_look_typhos, depression, irritability, muscle_pain, altered_sensorium,
    red_spots_over_body, belly_pain, abnormal_menstruation, dischromic_patches,
    watering_from_eyes, increased_appetite, polyuria, family_history, mucoid_sputum,
    rusty_sputum, lack_of_concentration, visual_disturbances, receiving_blood_transfusion,
    receiving_unsterile_injections, coma, stomach_bleeding, distention_of_abdomen,
    history_of_alcohol_consumption, fluid_overload_1, blood_in_sputum, prominent_veins_on_calf,
    palpitations, painful_walking, pus_filled_pimples, blackheads, scurring,
    skin_peeling, silver_like_dusting, small_dents_in_nails, inflammatory_nails, blister,
    red_sore_around_nose, yellow_crust_ooze
):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([[
        itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering,
        chills, joint_pain, stomach_pain, acidity, ulcers_on_tongue,
        muscle_wasting, vomiting, burning_micturition, spotting_urination, fatigue,
        weight_gain, anxiety, cold_hands_and_feets, mood_swings, weight_loss,
        restlessness, lethargy, patches_in_throat, irregular_sugar_level, cough,
        high_fever, sunken_eyes, breathlessness, sweating, dehydration,
        indigestion, headache, yellowish_skin, dark_urine, nausea, loss_of_appetite,
        pain_behind_the_eyes, back_pain, constipation, abdominal_pain, diarrhoea,
        mild_fever, yellow_urine, yellowing_of_eyes, acute_liver_failure, fluid_overload,
        swelling_of_stomach, swelled_lymph_nodes, malaise, blurred_and_distorted_vision,
        phlegm, throat_irritation, redness_of_eyes, sinus_pressure, runny_nose,
        congestion, chest_pain, weakness_in_limbs, fast_heart_rate, pain_during_bowel_movements,
        pain_in_anal_region, bloody_stool, irritation_in_anus, neck_pain, dizziness,
        cramps, bruising, obesity, swollen_legs, swollen_blood_vessels, puffy_face_and_eyes,
        enlarged_thyroid, brittle_nails, swollen_extremeties, excessive_hunger, extra_marital_contacts,
        drying_and_tingling_lips, slurred_speech, knee_pain, hip_joint_pain, muscle_weakness,
        stiff_neck, swelling_joints, movement_stiffness, spinning_movements, loss_of_balance,
        unsteadiness, weakness_of_one_body_side, loss_of_smell, bladder_discomfort,
        foul_smell_of_urine, continuous_feel_of_urine, passage_of_gases, internal_itching,
        toxic_look_typhos, depression, irritability, muscle_pain, altered_sensorium,
        red_spots_over_body, belly_pain, abnormal_menstruation, dischromic_patches,
        watering_from_eyes, increased_appetite, polyuria, family_history, mucoid_sputum,
        rusty_sputum, lack_of_concentration, visual_disturbances, receiving_blood_transfusion,
        receiving_unsterile_injections, coma, stomach_bleeding, distention_of_abdomen,
        history_of_alcohol_consumption, fluid_overload_1, blood_in_sputum, prominent_veins_on_calf,
        palpitations, painful_walking, pus_filled_pimples, blackheads, scurring,
        skin_peeling, silver_like_dusting, small_dents_in_nails, inflammatory_nails, blister,
        red_sore_around_nose, yellow_crust_ooze
    ]], columns=model.feature_names_in_)

    # Predict the disease
    prediction = model.predict(input_data)
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    return predicted_disease

# Define the inputs for Gradio
inputs = [
    gr.Checkbox(label='itching'), gr.Checkbox(label='skin_rash'),
    gr.Checkbox(label='nodal_skin_eruptions'), gr.Checkbox(label='continuous_sneezing'),
    gr.Checkbox(label='shivering'), gr.Checkbox(label='chills'),
    gr.Checkbox(label='joint_pain'), gr.Checkbox(label='stomach_pain'),
    gr.Checkbox(label='acidity'), gr.Checkbox(label='ulcers_on_tongue'),
    gr.Checkbox(label='muscle_wasting'), gr.Checkbox(label='vomiting'),
    gr.Checkbox(label='burning_micturition'), gr.Checkbox(label='spotting_urination'),
    gr.Checkbox(label='fatigue'), gr.Checkbox(label='weight_gain'),
    gr.Checkbox(label='anxiety'), gr.Checkbox(label='cold_hands_and_feets'),
    gr.Checkbox(label='mood_swings'), gr.Checkbox(label='weight_loss'),
    gr.Checkbox(label='restlessness'), gr.Checkbox(label='lethargy'),
    gr.Checkbox(label='patches_in_throat'), gr.Checkbox(label='irregular_sugar_level'),
    gr.Checkbox(label='cough'), gr.Checkbox(label='high_fever'),
    gr.Checkbox(label='sunken_eyes'), gr.Checkbox(label='breathlessness'),
    gr.Checkbox(label='sweating'), gr.Checkbox(label='dehydration'),
    gr.Checkbox(label='indigestion'), gr.Checkbox(label='headache'),
    gr.Checkbox(label='yellowish_skin'), gr.Checkbox(label='dark_urine'),
    gr.Checkbox(label='nausea'), gr.Checkbox(label='loss_of_appetite'),
    gr.Checkbox(label='pain_behind_the_eyes'), gr.Checkbox(label='back_pain'),
    gr.Checkbox(label='constipation'), gr.Checkbox(label='abdominal_pain'),
    gr.Checkbox(label='diarrhoea'), gr.Checkbox(label='mild_fever'),
    gr.Checkbox(label='yellow_urine'), gr.Checkbox(label='yellowing_of_eyes'),
    gr.Checkbox(label='acute_liver_failure'), gr.Checkbox(label='fluid_overload'),
    gr.Checkbox(label='swelling_of_stomach'), gr.Checkbox(label='swelled_lymph_nodes'),
    gr.Checkbox(label='malaise'), gr.Checkbox(label='blurred_and_distorted_vision'),
    gr.Checkbox(label='phlegm'), gr.Checkbox(label='throat_irritation'),
    gr.Checkbox(label='redness_of_eyes'), gr.Checkbox(label='sinus_pressure'),
    gr.Checkbox(label='runny_nose'), gr.Checkbox(label='congestion'),
    gr.Checkbox(label='chest_pain'), gr.Checkbox(label='weakness_in_limbs'),
    gr.Checkbox(label='fast_heart_rate'), gr.Checkbox(label='pain_during_bowel_movements'),
    gr.Checkbox(label='pain_in_anal_region'), gr.Checkbox(label='bloody_stool'),
    gr.Checkbox(label='irritation_in_anus'), gr.Checkbox(label='neck_pain'),
    gr.Checkbox(label='dizziness'), gr.Checkbox(label='cramps'),
    gr.Checkbox(label='bruising'), gr.Checkbox(label='obesity'),
    gr.Checkbox(label='swollen_legs'), gr.Checkbox(label='swollen_blood_vessels'),
    gr.Checkbox(label='puffy_face_and_eyes'), gr.Checkbox(label='enlarged_thyroid'),
    gr.Checkbox(label='brittle_nails'), gr.Checkbox(label='swollen_extremeties'),
    gr.Checkbox(label='excessive_hunger'), gr.Checkbox(label='extra_marital_contacts'),
    gr.Checkbox(label='drying_and_tingling_lips'), gr.Checkbox(label='slurred_speech'),
    gr.Checkbox(label='knee_pain'), gr.Checkbox(label='hip_joint_pain'),
    gr.Checkbox(label='muscle_weakness'), gr.Checkbox(label='stiff_neck'),
    gr.Checkbox(label='swelling_joints'), gr.Checkbox(label='movement_stiffness'),
    gr.Checkbox(label='spinning_movements'), gr.Checkbox(label='loss_of_balance'),
    gr.Checkbox(label='unsteadiness'), gr.Checkbox(label='weakness_of_one_body_side'),
    gr.Checkbox(label='loss_of_smell'), gr.Checkbox(label='bladder_discomfort'),
    gr.Checkbox(label='foul_smell_of_urine'), gr.Checkbox(label='continuous_feel_of_urine'),
    gr.Checkbox(label='passage_of_gases'), gr.Checkbox(label='internal_itching'),
    gr.Checkbox(label='toxic_look_typhos'), gr.Checkbox(label='depression'),
    gr.Checkbox(label='irritability'), gr.Checkbox(label='muscle_pain'),
    gr.Checkbox(label='altered_sensorium'), gr.Checkbox(label='red_spots_over_body'),
    gr.Checkbox(label='belly_pain'), gr.Checkbox(label='abnormal_menstruation'),
    gr.Checkbox(label='dischromic_patches'), gr.Checkbox(label='watering_from_eyes'),
    gr.Checkbox(label='increased_appetite'), gr.Checkbox(label='polyuria'),
    gr.Checkbox(label='family_history'), gr.Checkbox(label='mucoid_sputum'),
    gr.Checkbox(label='rusty_sputum'), gr.Checkbox(label='lack_of_concentration'),
    gr.Checkbox(label='visual_disturbances'), gr.Checkbox(label='receiving_blood_transfusion'),
    gr.Checkbox(label='receiving_unsterile_injections'), gr.Checkbox(label='coma'),
    gr.Checkbox(label='stomach_bleeding'), gr.Checkbox(label='distention_of_abdomen'),
    gr.Checkbox(label='history_of_alcohol_consumption'), gr.Checkbox(label='fluid_overload_1'),
    gr.Checkbox(label='blood_in_sputum'), gr.Checkbox(label='prominent_veins_on_calf'),
    gr.Checkbox(label='palpitations'), gr.Checkbox(label='painful_walking'),
    gr.Checkbox(label='pus_filled_pimples'), gr.Checkbox(label='blackheads'),
    gr.Checkbox(label='scurring'), gr.Checkbox(label='skin_peeling'),
    gr.Checkbox(label='silver_like_dusting'), gr.Checkbox(label='small_dents_in_nails'),
    gr.Checkbox(label='inflammatory_nails'), gr.Checkbox(label='blister'),
    gr.Checkbox(label='red_sore_around_nose'), gr.Checkbox(label='yellow_crust_ooze')
]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_disease,
    inputs=inputs,
    outputs='text',
    title='Disease Prediction Bot',
    description='Enter symptoms to get a disease prediction.'
)

# Launch the Gradio interface
interface.launch()
