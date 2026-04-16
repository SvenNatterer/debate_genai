import streamlit as st


def inject_arcade_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;800;900&display=swap');

        #MainMenu, header, footer {
            visibility: hidden;
        }

        html, body, [class*="css"] {
            font-family: 'Orbitron', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top, #1a1f35 0%, #0b0d14 55%, #05070c 100%);
            color: white;
        }

        .fighter-image {
            width: 100%;
            height: 260px;
            object-fit: cover;
            border-radius: 16px;
            margin-bottom: 12px;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        [data-testid="stButton"] > button {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.16);
            background: linear-gradient(180deg, rgba(48,68,120,0.95), rgba(22,30,58,0.95));
            color: white;
            font-weight: 800;
            letter-spacing: 0.5px;
            min-height: 48px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }

        [data-testid="stButton"] > button:hover {
            border-color: rgba(142,203,255,0.8);
            box-shadow: 0 0 0 1px rgba(142,203,255,0.25), 0 10px 24px rgba(0,0,0,0.35);
        }

        .arcade-title {
            text-align: center;
            font-size: 4rem;
            font-weight: 900;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            color: white;
            text-shadow: 0 0 10px rgba(255,255,255,0.15), 0 0 25px rgba(86,180,255,0.35);
        }

        .arcade-subtitle {
            text-align: center;
            font-size: 1rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            opacity: 0.8;
            margin-bottom: 1rem;
            color: #cfd8ff;
        }

        .arcade-panel {
            background: rgba(255,255,255,0.05);
            border: 2px solid rgba(255,255,255,0.14);
            border-radius: 22px;
            padding: 24px;
            box-shadow: 0 0 25px rgba(0,0,0,0.35);
            backdrop-filter: blur(10px);
        }

        .topic-panel {
            background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            border: 2px solid rgba(255,255,255,0.12);
            border-radius: 24px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 0 30px rgba(0,0,0,0.35);
        }

        .topic-label {
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 3px;
            opacity: 0.8;
            margin-bottom: 14px;
            color: #8ecbff;
            font-weight: 700;
        }

        .topic-text {
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.35;
            color: white;
        }

        .blink {
            animation: blink 1s steps(2, start) infinite;
        }

        .arcade-vs {
            text-align: center;
            font-size: 5rem;
            font-weight: 900;
            color: #ffd54a;
            text-shadow: 0 0 12px rgba(255,213,74,0.45);
            animation: pulse 1s infinite;
            margin-top: 80px;
        }

        .fighter-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            border: 2px solid rgba(255,255,255,0.10);
            border-radius: 20px;
            padding: 18px;
            min-height: 430px;
            text-align: center;
            box-shadow: 0 12px 28px rgba(0,0,0,0.25);
        }

        .fighter-image {
            width: 100%;
            height: 260px;
            object-fit: cover;
            border-radius: 16px;
            margin-bottom: 12px;
        }

        .fighter-name {
            font-size: 1.5rem;
            font-weight: 900;
            margin-top: 8px;
            text-transform: uppercase;
            color: white;
        }

        .fighter-side {
            font-size: 0.95rem;
            letter-spacing: 2px;
            opacity: 0.8;
            margin-top: 6px;
            color: #cfd8ff;
            text-transform: uppercase;
        }

        .fighter-stance {
            margin-top: 12px;
            font-size: 0.92rem;
            opacity: 0.8;
            line-height: 1.45;
            color: #e8ecff;
        }

        .score-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 14px;
        }

        .score-name {
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 10px;
            color: white;
        }

        .score-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(120px, 1fr));
            gap: 10px;
            color: #e8ecff;
            font-size: 0.95rem;
        }

        .winner-banner {
            background: linear-gradient(90deg, rgba(255,213,74,0.18), rgba(255,255,255,0.05));
            border: 1px solid rgba(255,213,74,0.35);
            border-radius: 18px;
            padding: 18px;
            font-size: 1.2rem;
            font-weight: 800;
            color: #fff3bf;
            margin-bottom: 1rem;
        }

        .small-status {
            text-align: center;
            color: #cfd8ff;
            opacity: 0.85;
            font-size: 0.9rem;
            letter-spacing: 1.2px;
            text-transform: uppercase;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.9; }
            50% { transform: scale(1.08); opacity: 1; }
            100% { transform: scale(1); opacity: 0.9; }
        }

        @keyframes blink {
            50% { opacity: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )