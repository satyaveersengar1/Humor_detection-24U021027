import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import Model  

st.set_page_config(page_title="Figure of Speech Detector", page_icon="🎭", layout="centered")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    
    model = Model(num_classes=5).to(device)
    model.load_state_dict(torch.load("saved_model.pth", map_location=device))
    model.eval()
    
    return tokenizer, model, device

tokenizer, model, device = load_model()

target_names = {0: "Sarcasm", 1: "Idiom", 2: "Irony", 3: "Metaphor", 4: "Hyperbole"}
class_colors = {"Sarcasm": "red", "Idiom": "blue", "Irony": "green", "Metaphor": "orange", "Hyperbole": "purple"}

st.title("🎭 Multimodal Figure of Speech Detector")
st.markdown("""
Welcome! Enter a sentence in Hindi, English or Hinglish along with its cultural context. 
The AI model will predict whether it's Sarcasm, Idiom, Irony, Metaphor, or Hyperbole.
""")

with st.form("prediction_form"):
    text_input = st.text_area("✍️ Enter Sentence (e.g., 'Bhopal mein itni garmi hai ki sadak par omelette ban jaye.')", height=100)
    context_input = st.text_input("🌍 Cultural Context (e.g., 'Extreme Indian summer heatwave')")
    
    submit_button = st.form_submit_button("Detect Figure of Speech 🚀")

if submit_button:
    if not text_input.strip():
        st.warning("Please enter a sentence to analyze.")
    else:
        with st.spinner("Model is analyzing..."):
            input_text = f"[CONTEXT: {context_input}] {text_input}"
            
            inputs = tokenizer(
                input_text,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            torch.manual_seed(42)
            visual_features = torch.randn(1, 512).to(device)
            
            with torch.no_grad():
                outputs = model(inputs["input_ids"], inputs["attention_mask"], visual_features)
                probabilities = F.softmax(outputs, dim=1)[0]
                predicted_class_id = torch.argmax(outputs, dim=1).item()
                
            predicted_label = target_names[predicted_class_id]
            confidence = probabilities[predicted_class_id].item() * 100
            
            st.success("Analysis Complete!")
            
            st.markdown(f"### Predicted Class: <span style='color:{class_colors[predicted_label]}'>**{predicted_label}**</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            st.markdown("#### Confidence Scores for all classes:")
            prob_dict = {target_names[i]: probabilities[i].item() for i in range(5)}
            st.bar_chart(prob_dict)

