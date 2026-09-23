import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from model.model import load_model
from utils.prediction import predict_image


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🖼️",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #888888;
        margin-bottom: 30px;
    }

    .info-card {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.10);
        border: 1px solid rgba(128, 128, 128, 0.20);
        text-align: center;
    }

    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(0, 128, 0, 0.10);
        border: 1px solid rgba(0, 128, 0, 0.25);
        margin-top: 15px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# PDF REPORT GENERATOR
# ==================================================

def generate_pdf_report(
    image,
    results,
    prediction_time
):

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]

    heading_style = styles["Heading2"]

    normal_style = styles["Normal"]

    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "AI Image Classification Report",
            title_style
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    # --------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "<b>Project:</b> AI Image Classification App",
            normal_style
        )
    )

    story.append(
        Paragraph(
            "<b>Model:</b> MobileNetV2",
            normal_style
        )
    )

    story.append(
        Paragraph(
            "<b>Dataset:</b> ImageNet",
            normal_style
        )
    )

    story.append(
        Paragraph(
            "<b>Classes:</b> 1000",
            normal_style
        )
    )

    story.append(
        Paragraph(
            "<b>Input Size:</b> 224 x 224",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Date & Time:</b> {prediction_time}",
            normal_style
        )
    )

    story.append(
        Spacer(
            1,
            20
        )
    )

    # --------------------------------------------------
    # IMAGE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Analyzed Image",
            heading_style
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    image_buffer = BytesIO()

    image_rgb = image.convert("RGB")

    image_rgb.save(
        image_buffer,
        format="JPEG"
    )

    image_buffer.seek(0)

    pdf_image = PDFImage(
        image_buffer,
        width=3.5 * inch,
        height=3.5 * inch
    )

    story.append(
        pdf_image
    )

    story.append(
        Spacer(
            1,
            20
        )
    )

    # --------------------------------------------------
    # MAIN PREDICTION
    # --------------------------------------------------

    top_result = results[0]

    main_label = top_result["label"]

    main_confidence = (
        top_result["confidence"] * 100
    )

    story.append(
        Paragraph(
            "Main Prediction",
            heading_style
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    prediction_table_data = [
        [
            "Prediction",
            "Confidence"
        ],
        [
            main_label,
            f"{main_confidence:.2f}%"
        ]
    ]

    prediction_table = Table(
        prediction_table_data,
        colWidths=[
            3.5 * inch,
            2 * inch
        ]
    )

    prediction_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.grey
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(
        prediction_table
    )

    story.append(
        Spacer(
            1,
            20
        )
    )

    # --------------------------------------------------
    # TOP 5 PREDICTIONS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Top 5 Predictions",
            heading_style
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    top5_data = [
        [
            "Rank",
            "Prediction",
            "Confidence"
        ]
    ]

    for index, result in enumerate(results):

        top5_data.append(
            [
                str(index + 1),
                result["label"],
                f'{result["confidence"] * 100:.2f}%'
            ]
        )

    top5_table = Table(
        top5_data,
        colWidths=[
            0.7 * inch,
            3.8 * inch,
            1.5 * inch
        ]
    )

    top5_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.grey
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(
        top5_table
    )

    story.append(
        Spacer(
            1,
            30
        )
    )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Generated by AI Image Classification App",
            normal_style
        )
    )

    story.append(
        Paragraph(
            "Powered by MobileNetV2 + TensorFlow",
            normal_style
        )
    )

    # --------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------

    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ==================================================
# TITLE
# ==================================================

st.markdown(
    '<div class="main-title">'
    '🖼️ AI Image Classification App'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an image or take a photo and let AI identify it.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def get_model():

    return load_model()


model = get_model()


# ==================================================
# PREDICTION HISTORY
# ==================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ==================================================
# MODEL INFORMATION
# ==================================================

st.markdown(
    '<div class="section-title">'
    '🧠 Model Information'
    '</div>',
    unsafe_allow_html=True
)


info1, info2, info3, info4 = st.columns(4)


with info1:

    st.markdown(
        """
        <div class="info-card">

        <h3>🤖 Model</h3>

        <p>MobileNetV2</p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
        <div class="info-card">

        <h3>📚 Dataset</h3>

        <p>ImageNet</p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
        <div class="info-card">

        <h3>🔢 Classes</h3>

        <p>1000</p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info4:

    st.markdown(
        """
        <div class="info-card">

        <h3>📐 Input Size</h3>

        <p>224 × 224</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# IMAGE INPUT
# ==================================================

st.markdown(
    '<div class="section-title">'
    '📤 Select Image Input'
    '</div>',
    unsafe_allow_html=True
)


input_method = st.radio(
    "Choose how you want to provide the image:",
    [
        "📁 Upload Image",
        "📷 Take Photo"
    ],
    horizontal=True
)


uploaded_file = None

camera_image = None


# ==================================================
# UPLOAD IMAGE
# ==================================================

if input_method == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# ==================================================
# CAMERA INPUT
# ==================================================

elif input_method == "📷 Take Photo":

    camera_image = st.camera_input(
        "Take a photo using your camera"
    )


# ==================================================
# GET SELECTED IMAGE
# ==================================================

image = None


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )


elif camera_image is not None:

    image = Image.open(
        camera_image
    )


# ==================================================
# DISPLAY IMAGE
# ==================================================

if image is not None:

    if image.mode not in ["RGB", "RGBA"]:

        image = image.convert("RGB")


    left, right = st.columns(
        [1.2, 1]
    )


    # ==================================================
    # LEFT SIDE
    # ==================================================

    with left:

        st.subheader(
            "📷 Selected Image"
        )

        st.image(
            image,
            caption="Image for classification",
            use_container_width=True
        )


    # ==================================================
    # RIGHT SIDE
    # ==================================================

    with right:

        st.subheader(
            "🔍 Classification"
        )

        st.write(
            "Click the button below to analyze "
            "the image using the MobileNetV2 AI model."
        )


        classify_button = st.button(
            "🔍 Classify Image",
            type="primary",
            use_container_width=True
        )


        # ==================================================
        # CLASSIFY IMAGE
        # ==================================================

        if classify_button:

            try:

                with st.spinner(
                    "🤖 AI is analyzing the image..."
                ):

                    results = predict_image(
                        model,
                        image
                    )

            except Exception:

                st.error(
                    "❌ Unable to classify this image."
                )

                st.warning(
                    "Please try another JPG or PNG image."
                )

                st.stop()


            # ==================================================
            # TOP PREDICTION
            # ==================================================

            top_result = results[0]

            label = top_result["label"]

            confidence = top_result["confidence"]


            # ==================================================
            # SAVE TO HISTORY
            # ==================================================

            history_item = {

                "Time": datetime.now().strftime(
                    "%H:%M:%S"
                ),

                "Prediction": label,

                "Confidence (%)": round(
                    confidence * 100,
                    2
                )
            }


            st.session_state.prediction_history.append(
                history_item
            )


            # ==================================================
            # PREDICTION CARD
            # ==================================================

            st.markdown(
                '<div class="prediction-card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "🎯 Prediction"
            )

            st.success(
                f"{label} — "
                f"{confidence * 100:.2f}% confidence"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


            # ==================================================
            # TOP 5 PREDICTIONS
            # ==================================================

            st.subheader(
                "🏆 Top 5 Predictions"
            )


            for index, result in enumerate(
                results
            ):

                result_label = result[
                    "label"
                ]

                result_confidence = result[
                    "confidence"
                ]


                st.write(
                    f"**{index + 1}. "
                    f"{result_label}**"
                )


                st.progress(
                    result_confidence,
                    text=(
                        f"{result_confidence * 100:.2f}%"
                    )
                )


            # ==================================================
            # CONFIDENCE CHART
            # ==================================================

            st.subheader(
                "📊 Prediction Confidence Chart"
            )


            chart_data = {}


            for result in results:

                result_label = result[
                    "label"
                ]

                result_confidence = (
                    result["confidence"] * 100
                )


                chart_data[
                    result_label
                ] = result_confidence


            st.bar_chart(
                chart_data,
                x_label="Prediction",
                y_label="Confidence (%)"
            )


            st.caption(
                "The chart shows the confidence "
                "scores for the model's Top 5 predictions."
            )


            # ==================================================
            # CSV DOWNLOAD
            # ==================================================

            st.subheader(
                "📥 Download Prediction Results"
            )


            download_data = []


            for index, result in enumerate(
                results
            ):

                download_data.append(
                    {
                        "Rank": index + 1,

                        "Prediction": result[
                            "label"
                        ],

                        "Confidence (%)": round(
                            result[
                                "confidence"
                            ] * 100,
                            2
                        )
                    }
                )


            results_df = pd.DataFrame(
                download_data
            )


            csv_data = results_df.to_csv(
                index=False
            )


            st.download_button(
                label="⬇️ Download Results as CSV",

                data=csv_data,

                file_name="prediction_results.csv",

                mime="text/csv",

                use_container_width=True
            )


            # ==================================================
            # RESULTS TABLE
            # ==================================================

            st.write(
                "Prediction results:"
            )


            st.dataframe(
                results_df,

                use_container_width=True,

                hide_index=True
            )


            # ==================================================
            # PDF REPORT
            # ==================================================

            st.subheader(
                "📄 Download PDF Report"
            )


            prediction_time = datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )


            pdf_data = generate_pdf_report(
                image=image,
                results=results,
                prediction_time=prediction_time
            )


            st.download_button(
                label="📄 Download Prediction Report",

                data=pdf_data,

                file_name="AI_Image_Classification_Report.pdf",

                mime="application/pdf",

                use_container_width=True
            )


# ==================================================
# PREDICTION HISTORY
# ==================================================

if st.session_state.prediction_history:

    st.markdown("---")

    st.subheader(
        "🕘 Prediction History"
    )


    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )


    st.dataframe(
        history_df,

        use_container_width=True,

        hide_index=True
    )


    # ==================================================
    # CLEAR HISTORY
    # ==================================================

    if st.button(
        "🗑️ Clear Prediction History"
    ):

        st.session_state.prediction_history = []

        st.rerun()


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")


st.markdown(
    """
    <div style="text-align:center; color:#888888;">

    <p>
    🧠 Powered by MobileNetV2 + TensorFlow
    </p>

    <p>
    📷 Upload + Camera AI Image Classification
    </p>

    <p>
    📄 Automated PDF Prediction Reports
    </p>

    <p>
    Image Classification App | Final Year CSE Project
    </p>

    </div>
    """,
    unsafe_allow_html=True
)