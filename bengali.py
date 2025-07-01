import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer, util
import torch

# --- Google Sheets Auth ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
client = gspread.authorize(creds)
sheet = client.open("Bengal-Art-Ground").sheet1  # Replace with your exact sheet name

# --- Load AI Model ---
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",device="cpu")

# --- Tagging Keywords ---
tags_dict = {
    "ভালোবাসা": "Love",
    "দুঃখ": "Sadness",
    "রাত্রি": "Night",
    "স্বপ্ন": "Dream",
    "প্রকৃতি": "Nature",
    "আনন্দ": "Joy",
    "যুদ্ধ": "War"
}

def auto_tag(text):
    tags = []
    for word, tag in tags_dict.items():
        if word in text:
            tags.append(tag)
    return ", ".join(tags) if tags else "Other"

# --- Streamlit UI ---
st.image("logo.jpg", width=150)  # Optional logo
st.title("🎨 Bengal Art Ground")
st.markdown("""
<style>
h1 {
    color: #800000;
}
textarea, input {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.subheader("📬 Submit Your Poem or Story")
st.markdown("---")

# --- Input Form ---
title = st.text_input("📝 শিরোনাম (Title)")
author = st.text_input("👤 আপনার নাম (Author)")
content = st.text_area("✍️ আপনার কবিতা বা গল্প লিখুন (Poem or Story)")

# --- Submit Logic ---
if st.button("Submit"):
    if title and author and content:
        # Save to Google Sheet
        sheet.append_row([title, author, content, ""])  # Reserve 4th col for tags
        st.success("✅ সফলভাবে জমা হয়েছে! (Submitted)")

        # Auto-tagging
        tag_result = auto_tag(content)
        last_row = len(sheet.get_all_values())
        sheet.update_cell(last_row, 4, tag_result)

        # Similarity-based recommendations
        existing_posts = sheet.get_all_records()
        existing_texts = [row['Content'] for row in existing_posts if row['Content'] != content]

        if existing_texts:
            new_embedding = model.encode(content, convert_to_tensor=True)
            existing_embeddings = model.encode(existing_texts, convert_to_tensor=True)
            cos_sim = util.pytorch_cos_sim(new_embedding, existing_embeddings)[0]
            top_indices = torch.topk(cos_sim, k=2).indices

            st.markdown("### 🔁 You may also like:")
            for idx in top_indices:
                similar_post = existing_posts[idx]
                st.markdown(f"**📝 {similar_post['Title']}** by {similar_post['Author']}")
                st.write(similar_post["Content"])
                st.markdown("---")
    else:
        st.warning("⚠️ সব ঘর পূরণ করুন। (Please fill all fields)")

# --- Display All Posts ---
st.markdown("---")
st.subheader("📖 Explore Submissions")

data = sheet.get_all_records()

# Filters
authors = sorted(list(set(row["Author"] for row in data)))
titles = sorted(list(set(row["Title"] for row in data)))

st.sidebar.title("🔍 Filter Submissions")
author_filter = st.sidebar.selectbox("Filter by Author", ["All"] + authors)
title_filter = st.sidebar.selectbox("Filter by Title", ["All"] + titles)

filtered_data = data
if author_filter != "All":
    filtered_data = [row for row in filtered_data if row["Author"] == author_filter]
if title_filter != "All":
    filtered_data = [row for row in filtered_data if row["Title"] == title_filter]

# Display posts
for row in filtered_data:
    st.markdown(f"### 📝 {row['Title']}")
    st.markdown(f"**👤 {row['Author']}**")
    st.write(f"📖 {row['Content']}")

    if 'Tags' in row and row['Tags']:
        st.markdown(f"🏷️ Tags: *{row['Tags']}*")

    st.markdown("---")
    st.markdown("---")
st.markdown("Made with ❤️ by a medical student using Streamlit + AI")

