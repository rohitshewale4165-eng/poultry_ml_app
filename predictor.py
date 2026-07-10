import frappe
import joblib
import json
import os

def get_model_path(filename):
    return os.path.join(frappe.get_app_path("poultry_ml_app"), "ml_models", filename)

@frappe.whitelist()
def predict_poultry_erp_final(shed_length, shed_width, side_height, centre_height,
                               num_sheds, bird_density, extra_charges, other_charges,
                               electric_phase, equipment_brand, material_quality, equipment_cost):

    shed_length = float(shed_length)
    shed_width = float(shed_width)
    side_height = float(side_height)
    centre_height = float(centre_height)
    num_sheds = int(num_sheds)
    bird_density = float(bird_density)
    extra_charges = float(extra_charges)
    other_charges = float(other_charges)
    equipment_cost = float(equipment_cost)

    if num_sheds <= 0:
        frappe.throw("num_sheds must be greater than 0.")
    if shed_length <= 0 or shed_width <= 0:
        frappe.throw("shed_length and shed_width must be greater than 0.")

    model = joblib.load(get_model_path("erp_final_model.pkl"))
    recommendation_model = joblib.load(get_model_path("erp_recommendation_classifier.pkl"))
    anomaly_detector = joblib.load(get_model_path("erp_anomaly_detector.pkl"))

    with open(get_model_path("erp_final_enc.json"), "r") as f:
        encs = json.load(f)
    with open(get_model_path("erp_model_metrics.json"), "r") as f:
        metrics = json.load(f)
    with open(get_model_path("erp_top_feature.json"), "r") as f:
        top_feature_data = json.load(f)

    top_cost_drivers_list = top_feature_data.get("top_features", [])
    formatted_drivers = []
    for i, feature_name in enumerate(top_cost_drivers_list):
        formatted_drivers.append(f"{i+1}. {feature_name.replace('_', ' ').title().replace('Enc', '')}")
    top_cost_driver_formatted = " | ".join(formatted_drivers) if formatted_drivers else "N/A"

    total_area = (shed_length * shed_width) * num_sheds
    avg_height = (side_height + centre_height) / 2
    total_birds = total_area / bird_density if bird_density > 0 else 0
    birds_per_shed = total_birds / num_sheds

    area_per_bird = total_area / total_birds if total_birds > 0 else 0
    cost_per_sqft_equipment = equipment_cost / total_area if total_area > 0 else 0
    height_x_area = avg_height * total_area

    most_frequent = encs["_most_frequent"]
    fallback_messages_list = []

    if electric_phase not in encs["Electric_Phase"]:
        fallback_messages_list.append(f"Electric_Phase value '{electric_phase}' not recognized, defaulted to most common category ('{most_frequent['Electric_Phase']}')")
    p_enc = encs["Electric_Phase"].get(electric_phase, encs["Electric_Phase"][most_frequent["Electric_Phase"]])

    if equipment_brand not in encs["Equipment_Brand"]:
        fallback_messages_list.append(f"Equipment_Brand value '{equipment_brand}' not recognized, defaulted to most common category ('{most_frequent['Equipment_Brand']}')")
    b_enc = encs["Equipment_Brand"].get(equipment_brand, encs["Equipment_Brand"][most_frequent["Equipment_Brand"]])

    if material_quality not in encs["Material_Quality"]:
        fallback_messages_list.append(f"Material_Quality value '{material_quality}' not recognized, defaulted to most common category ('{most_frequent['Material_Quality']}')")
    q_enc = encs["Material_Quality"].get(material_quality, encs["Material_Quality"][most_frequent["Material_Quality"]])

    category_fallback_used_message = ", ".join(fallback_messages_list) if fallback_messages_list else "None"

    X = [[total_area, bird_density, total_birds, avg_height, p_enc, b_enc, q_enc,
          equipment_cost, other_charges,
          area_per_bird, cost_per_sqft_equipment, height_x_area]]

    raw_prediction_unfloored =float(model.predict(X)[0])
    min_cost_floor = total_area * 325
    base_cost = max(raw_prediction_unfloored, min_cost_floor)
    floor_applied = (min_cost_floor > raw_prediction_unfloored)

    anomaly_score =int(anomaly_detector.predict(X)[0])
    is_anomaly = True if anomaly_score == -1 else False
    # unusual_input_warning = "Yes - This input is significantly different from typical sheds in our training data. Prediction reliability may be reduced." if is_anomaly else "No"

    residual_std = metrics["residual_std"]
    r2_score_value = metrics["r2_score"]
    relative_error = residual_std / base_cost if base_cost > 0 else 1
    ai_confidence = max(0, min(100, (1 - relative_error) * r2_score_value * 100))
    margin_95 = 1.96 * residual_std

    predicted_recommendation =str(recommendation_model.predict(X)[0])

    total_install = base_cost + extra_charges
    grand_total = total_install + equipment_cost + other_charges
    cpb = grand_total / total_birds if total_birds > 0 else 0
    cpsf = grand_total / total_area if total_area > 0 else 0

    return {
        "Average Height": round(avg_height, 2),
        "Total Area (Sq Ft)": total_area,
        "Total Birds Per Shed": round(birds_per_shed, 0),
        "Total Birds All Sheds": round(total_birds, 0),
        "AI Predicted Cost": round(base_cost, 2),
        "Extra Charges": extra_charges,
        "Total Installation Cost": round(total_install, 2),
        "Category Fallback Used": category_fallback_used_message,
        "Floor Applied": floor_applied,
        # "Unusual Input Warning": unusual_input_warning,
        "Raw AI Prediction": round(raw_prediction_unfloored, 2),
        "AI Confidence %": round(ai_confidence, 1),
        "Prediction Range (95%)": f"₹{base_cost - margin_95:,.0f} - ₹{base_cost + margin_95:,.0f}",
        "AI Recommendation": predicted_recommendation,
        "Top Cost Driver": top_cost_driver_formatted,
        "Status": "Success",
        "Equipment Total": equipment_cost,
        "Installation Total": round(total_install, 2),
        "Other Charges": other_charges,
        "Grand Total": round(grand_total, 2),
        "Cost Per Bird": round(cpb, 2),
        "Cost Per Sq Ft": round(cpsf, 2)
    }











