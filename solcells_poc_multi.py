

import streamlit as st
import requests
import random
import pandas as pd
import numpy as np
import hashlib
from math import radians, cos, sin
import pydeck as pdk
import time
import os
import re

# -----------------------------
# Konstanter / antaganden för mock
# -----------------------------
USER_AGENT = "SolcellsPOC/1.3 (isak.gunnardo@hotmail.com)"  # <-- BYT till din riktiga kontaktadress
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
SE_LATLON_IRRADIANCE_KWH_PER_KWP = 1000  # grovt snitt Sverige
M2_PER_KWP = 6.5                         # ~6–7 m² per kWp
DEFAULT_ELECTRICITY_PRICE = 1.5          # kr/kWh (ändras i UI)
MAX_POINTS = 120                         # begränsa antal rooftops i generering

# Fallback-API (valfritt): sätt som env-variabler i din miljö
OPENCAGE_KEY = os.getenv("8c6d64f166f14eaa91394c82737eb1b7")   # https://opencagedata.com
GOOGLE_KEY = os.getenv("AIzaSyDlzq38XW1X4j8-pMEenq97XgFOlqQXgkc") # https://developers.google.com/maps/documentation/geocoding

# Minimal gazetteer-fallback för demo/offline
GAZETTEER_SE = {
    "lund": (55.70466, 13.19101, "Lund, Skåne, Sverige"),
    "växjö": (56.8790, 14.8059, "Växjö, Kronoberg, Sverige"),
    "goteborg": (57.70887, 11.97456, "Göteborg, Västra Götaland, Sverige"),
    "göteborg": (57.70887, 11.97456, "Göteborg, Västra Götaland, Sverige"),
    "stockholm": (59.3293, 18.0686, "Stockholm, Sverige"),
    "malmö": (55.6050, 13.0038, "Malmö, Skåne, Sverige"),
    "malmo": (55.6050, 13.0038, "Malmö, Skåne, Sverige"),
    "jönköping": (57.7815, 14.1562, "Jönköping, Sverige"),
    "vetlanda": (57.4275, 15.0853, "Vetlanda, Sverige"),
}

# -----------------------------
# Hjälp: deterministisk seed per ort/postnr
# -----------------------------
def seed_from_text(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

# -----------------------------
# 1. Geocode location – robust + debug + fallbacks
# -----------------------------
def geocode_location(location: str):
    """
    Geokodar med Nominatim. Om det failar:
      - försöker OpenCage om OPENCAGE_API_KEY finns
      - försöker Google Geocoding om GOOGLE_GEOCODING_KEY finns
      - som sista fallback använder en liten svensk gazetteer (mock)
    Returnerar (lat, lon, display_name) eller (None, None, None)
    """
    lat, lon, name, diag = geocode_with_nominatim(location)
    if lat is not None:
        return lat, lon, name

    # Fallback: OpenCage
    if OPENCAGE_KEY:
        lat, lon, name, oc_diag = geocode_with_opencage(location, OPENCAGE_KEY)
        if lat is not None:
            return lat, lon, name
        diag += f" | OpenCage: {oc_diag}"

    # Fallback: Google Geocoding
    if GOOGLE_KEY:
        lat, lon, name, g_diag = geocode_with_google(location, GOOGLE_KEY)
        if lat is not None:
            return lat, lon, name
        diag += f" | Google: {g_diag}"

    # Sista fallback: liten inbyggd lista (mock/offline)
    lat, lon, name = geocode_with_gazetteer(location)
    if lat is not None:
        return lat, lon, name

    # Om allt misslyckas – logga i Streamlit för felsökning
    st.session_state["geocode_diag"] = diag
    return None, None, None

def geocode_with_nominatim(location: str):
    if not location or not location.strip():
        return None, None, None, "empty query"

    params = {
        "q": location,
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "se",
        "addressdetails": 0,
        "email": "isak.gunnardo@hotmail.com",  # <-- BYT till din riktiga e-post
    }
    headers = {
        "User-Agent": USER_AGENT,
        "accept-language": "sv-SE,sv;q=0.9,en;q=0.8",
    }

    diag = ""
    for attempt in range(3):
        try:
            resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    display_name = data[0].get("display_name", location)
                    return lat, lon, display_name, "ok"
                else:
                    return None, None, None, "nominatim: no results"
            elif resp.status_code in (429, 403, 503):
                diag = f"nominatim: {resp.status_code}"
                time.sleep(1 + attempt)  # backoff
            else:
                return None, None, None, f"nominatim: http {resp.status_code}"
        except requests.RequestException as e:
            diag = f"nominatim exception: {e}"
            time.sleep(1 + attempt)

    return None, None, None, diag or "nominatim: unknown error"

def geocode_with_opencage(location: str, key: str):
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": location, "key": key, "limit": 1, "language": "sv", "countrycode": "se"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            j = r.json()
            if j.get("results"):
                g = j["results"][0]["geometry"]
                lat, lon = g["lat"], g["lng"]
                name = j["results"][0].get("formatted", location)
                return lat, lon, name, "ok"
            return None, None, None, "opencage: no results"
        return None, None, None, f"opencage: http {r.status_code}"
    except requests.RequestException as e:
        return None, None, None, f"opencage exception: {e}"

def geocode_with_google(location: str, key: str):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location, "key": key, "language": "sv"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            j = r.json()
            if j.get("results"):
                g = j["results"][0]["geometry"]["location"]
                lat, lon = g["lat"], g["lng"]
                name = j["results"][0].get("formatted_address", location)
                return lat, lon, name, "ok"
            return None, None, None, f"google: {j.get('status', 'no results')}"
        return None, None, None, f"google: http {r.status_code}"
    except requests.RequestException as e:
        return None, None, None, f"google exception: {e}"

def geocode_with_gazetteer(location: str):
    key = location.strip().lower()
    key = re.sub(r"\s+", "", key)
    if key in GAZETTEER_SE:
        lat, lon, name = GAZETTEER_SE[key]
        return lat, lon, name
    # Enkel postnummer-match: om bara siffror typ 5 tecken – mocka Stockholm
    if re.fullmatch(r"\d{3}\s?\d{2}", location.strip()):
        return 59.3293, 18.0686, f"Postnummer {location} (mock, Stockholm)"
    return None, None, None

# -----------------------------
# 2. Mock image fetch (plats för riktiga integrationer)
# -----------------------------
def fetch_planet_image(lat, lon):
    return "https://placehold.co/800x450?text=Planet+Labs+Image"

# -----------------------------
# 3. Feature engineering (mock)
# -----------------------------
def pv_kwp_from_area(m2: float) -> float:
    return max(0.0, m2 / M2_PER_KWP)

def annual_kwh(kWp: float) -> float:
    return kWp * SE_LATLON_IRRADIANCE_KWH_PER_KWP

def annual_savings_ksek(annual_kwh_value: float, price_per_kwh: float) -> float:
    return (annual_kwh_value * price_per_kwh) / 1000.0  # kSEK

def random_point_around(lat, lon, max_distance_m=500):
    # meters to degrees ungefärligt
    dlat = (random.uniform(-1, 1) * max_distance_m) / 111_320
    dlon = (random.uniform(-1, 1) * max_distance_m) / (40075000 * cos(radians(lat)) / 360)
    return lat + dlat, lon + dlon

# -----------------------------
# 4. Lead scoring (förbättrad)
# -----------------------------
def calculate_lead_score(area_m2, income, has_ev, shadow_index, roof_age, est_savings_ksek, has_pool=False, is_farm=False, high_tariff=False, heating_type="Direktel"):
    score = 0.0
    
    # Baspoäng från takarea
    if area_m2 >= 25:
        score += min(40, (area_m2 - 20) * 0.8)
    
    # Inkomstbonus
    if income >= 600_000:
        score += 15
        if income >= 750_000:  # Från bildens krav
            score += 10
        if income >= 900_000:
            score += 5
    
    # Elbil boost
    if has_ev:
        score += 10
    
    # Skuggindex
    score += max(0, (1 - shadow_index)) * 15
    
    # Takålder
    if 18 <= roof_age <= 30:
        score += 8
    elif roof_age < 8:
        score -= 5
    
    # Ekonomisk potential
    if est_savings_ksek >= 7.5:
        score += 10
    elif est_savings_ksek >= 4:
        score += 6
    elif est_savings_ksek >= 2:
        score += 3
    
    # NYA FAKTORER FRÅN BILDEN:
    
    # Pool/Jacuzzi = högre elförbrukning
    if has_pool:
        score += 12
    
    # Lantbruk/Gård = större tak, mer potential
    if is_farm:
        score += 8
    
    # Höga eltariffer = större motivation
    if high_tariff:
        score += 15
    
    # Uppvärmningstyp påverkar
    if heating_type == "Direktel":
        score += 8  # Mest att spara
    elif heating_type == "Bergvärme":
        score += 5  # Bra kombination med solceller
    elif heating_type in ["Fjärrvärme", "Pellets"]:
        score += 2
    
    return int(max(0, min(100, round(score))))

# -----------------------------
# 5. Generera rooftops (mock) med adresser/latlon
# -----------------------------
def generate_rooftop_data(location, center_lat, center_lon, n, price_per_kwh, include_with_panels=False):
    streets = ["Solgatan", "Energivägen", "Panelgränd", "Takstigen", "Ljusallén", "Voltvägen", "Amperegränd"]
    rooftops = []
    for i in range(n):
        lat, lon = random_point_around(center_lat, center_lon, max_distance_m=random.choice([250, 350, 500]))
        area = round(random.uniform(18, 120), 1)
        pitch = random.choice([10, 20, 30, 35, 40, 45])
        azimuth = random.choice(["S", "SV", "SO", "V", "O"])
        has_panels = random.random() < 0.25
        income = random.randint(400_000, 1_200_000)
        household_size = random.randint(1, 5)
        has_ev = random.random() < 0.32
        owner_age = random.randint(28, 78)
        roof_age = random.randint(1, 35)
        shadow_index = round(random.uniform(0.0, 1.0), 2)
        consumption_kwh = random.choice([9000, 12000, 15000, 20000, 25000])
        
        # Nya parametrar från bildens krav
        has_pool = random.random() < 0.15  # 15% har pool/jacuzzi
        is_farm = random.random() < 0.08   # 8% lantbruk/gårdar
        roof_type = random.choice(["Tegel", "Plåt", "Betong", "Papp"])
        heating_type = random.choice(["Bergvärme", "Direktel", "Fjärrvärme", "Pellets", "Olja"])
        tariff_type = random.choice(["Fast", "Rörlig", "Effekt+", "Tidsdiff"])
        high_tariff = tariff_type in ["Effekt+", "Tidsdiff"] and consumption_kwh > 15000

        kWp = round(pv_kwp_from_area(area), 2)
        kwh_year = round(annual_kwh(kWp))
        savings_ksek = round(annual_savings_ksek(kwh_year, price_per_kwh), 1)

        score = calculate_lead_score(
            area_m2=area,
            income=income,
            has_ev=has_ev,
            shadow_index=shadow_index,
            roof_age=roof_age,
            est_savings_ksek=savings_ksek,
            has_pool=has_pool,
            is_farm=is_farm,
            high_tariff=high_tariff,
            heating_type=heating_type
        )

        address = f"{random.choice(streets)} {random.randint(1, 82)}, {location}"

        rooftops.append({
            "Tak-ID": f"Tak-{i+1}",
            "Adress": address,
            "Lat": lat,
            "Lon": lon,
            "Area (m²)": area,
            "Lutning (°)": pitch,
            "Väderstreck": azimuth,
            "Taktyp": roof_type,
            "Solpaneler": "Ja" if has_panels else "Nej",
            "Takålder (år)": roof_age,
            "Skuggindex (0-1)": shadow_index,
            "Inkomst (SEK/år)": income,
            "Hushållsstorlek": household_size,
            "Pool/Jacuzzi": "Ja" if has_pool else "Nej",
            "Lantbruk/Gård": "Ja" if is_farm else "Nej",
            "Uppvärmning": heating_type,
            "Eltariff": tariff_type,
            "Hög tariff": "Ja" if high_tariff else "Nej",
            "Elbil": "Ja" if has_ev else "Nej",
            "Elförbrukning (kWh/år)": consumption_kwh,
            "kWp (est)": kWp,
            "kWh/år (est)": kwh_year,
            "Besparing (kSEK/år)": savings_ksek,
            "Lead Score": score,
        })

    df = pd.DataFrame(rooftops)
    if not include_with_panels:
        df = df[df["Solpaneler"] == "Nej"].reset_index(drop=True)
    return df

# -----------------------------
# 6. UI / Dashboard
# -----------------------------
def dashboard():
    st.set_page_config(page_title="Solcells POC", page_icon="🔆", layout="wide")
    st.title("🔆 Solcells POC – Kartvy & Lead Scoring")

    with st.sidebar:
        st.header("Sökning")
        location = st.text_input("Ange ort eller postnummer", value="")
        st.caption("Tips: Testa en svensk ort, t.ex. 'Lund', 'Växjö', '114 28' etc.")
        price_per_kwh = st.number_input("Elpris (kr/kWh)", min_value=0.0, step=0.1, value=DEFAULT_ELECTRICITY_PRICE)
        n_points = st.slider("Antal hushåll (mock)", 10, MAX_POINTS, 60, step=10)
        include_panels = st.checkbox("Inkludera hushåll som redan har paneler", value=False)

        st.header("Filter")
        min_area = st.slider("Minsta takarea (m²)", 0, 120, 25)
        min_score = st.slider("Minsta lead score", 0, 100, 50)
        min_income = st.slider("Minsta inkomst (kSEK)", 400, 1200, 750)
        only_ev = st.checkbox("Endast hushåll med elbil")
        only_pool = st.checkbox("Endast hushåll med pool/jacuzzi")
        only_high_tariff = st.checkbox("Endast höga eltariffer")
        heating_filter = st.selectbox("Uppvärmningstyp", ["Alla", "Direktel", "Bergvärme", "Fjärrvärme", "Pellets", "Olja"])

        sort_by = st.selectbox("Sortera efter", ["Lead Score", "Besparing (kSEK/år)", "Area (m²)", "kWh/år (est)"])
        sort_ascending = st.checkbox("Stigande sortering", value=False)

        st.header("Avancerat")
        show_diag = st.checkbox("Visa geokodningsdiagnostik", value=False)

    if not location:
        st.info("👈 Ange en ort eller ett postnummer för att starta analysen.")
        return

    with st.spinner("Geokodar plats…"):
        lat, lon, display_name = geocode_location(location)

    if not lat or not lon:
        st.error("Kunde inte geokoda platsen.")
        st.caption("Tips: Ange en mer specifik ort eller postnummer (t.ex. '114 28' eller 'Växjö').")
        if show_diag:
            diag = st.session_state.get("geocode_diag", "(ingen diagnos tillgänglig)")
            st.code(diag)
        # Demonstrations-fallback: kör ändå vidare på Stockholm (så UI kan testas)
        if st.toggle("Använd mock-position (Stockholm) för demo", value=True):
            lat, lon, display_name = 59.3293, 18.0686, "Stockholm (mock)"
        else:
            return

    # Deterministisk seed baserat på inmatning
    random.seed(seed_from_text(location))

    # Bilder (mock)
    st.image(fetch_planet_image(lat, lon), caption="Planet Labs Satellitbild (mock)")

    # Generera data
    df = generate_rooftop_data(
        location=display_name or location,
        center_lat=lat,
        center_lon=lon,
        n=n_points,
        price_per_kwh=price_per_kwh,
        include_with_panels=include_panels
    )

    # Filtrering
    mask = (df["Area (m²)"] >= min_area) & (df["Lead Score"] >= min_score) & (df["Inkomst (SEK/år)"] >= min_income * 1000)
    if only_ev:
        mask &= (df["Elbil"] == "Ja")
    if only_pool:
        mask &= (df["Pool/Jacuzzi"] == "Ja")
    if only_high_tariff:
        mask &= (df["Hög tariff"] == "Ja")
    if heating_filter != "Alla":
        mask &= (df["Uppvärmning"] == heating_filter)
    df_filtered = df[mask].copy()

    # Sortering
    df_filtered.sort_values(by=sort_by, ascending=sort_ascending, inplace=True)

    st.subheader(f"Analys för: {display_name or location}")
    st.caption(f"Centroid: {lat:.5f}, {lon:.5f} — Antal hushåll (efter filter): {len(df_filtered)}")

    # Karta (pydeck)
    if not df_filtered.empty:
        color_by_score = df_filtered["Lead Score"].apply(lambda s: [int(255*(s/100)), int(255*(1 - s/100)), 80])
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered.assign(color=color_by_score),
            get_position=["Lon", "Lat"],
            get_radius=6,
            get_fill_color="color",
            pickable=True,
        )
        tooltip = {
            "html": "<b>{Adress}</b><br/>Score: {Lead Score}<br/>Area: {Area (m²)} m²<br/>Besparing: {Besparing (kSEK/år)} kSEK<br/>Pool: {Pool/Jacuzzi}<br/>Elbil: {Elbil}<br/>Tariff: {Eltariff}",
            "style": {"backgroundColor": "white", "color": "black"}
        }
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
    else:
        st.warning("Inga hushåll matchade filtren.")

    # Tabell
    st.markdown("### Resultattabell")
    st.dataframe(
        df_filtered[
            [
                "Tak-ID", "Adress", "Area (m²)", "Lutning (°)", "Väderstreck", "Taktyp",
                "Skuggindex (0-1)", "Takålder (år)", "Pool/Jacuzzi", "Lantbruk/Gård", 
                "Uppvärmning", "Eltariff", "Hög tariff", "Elbil", "Inkomst (SEK/år)",
                "kWp (est)", "kWh/år (est)", "Besparing (kSEK/år)", "Lead Score"
            ]
        ],
        use_container_width=True
    )

    # Top leads
    st.markdown("### Top 5 leads")
    top_leads = df_filtered.sort_values(by="Lead Score", ascending=False).head(5)
    st.table(top_leads[["Tak-ID", "Adress", "Lead Score", "Besparing (kSEK/år)", "Area (m²)", "kWh/år (est)"]])

    # Export
    col1, col2 = st.columns(2)
    with col1:
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Ladda ner CSV", data=csv, file_name="solcells_leads.csv", mime="text/csv")
    with col2:
        if st.button("🚀 Skicka till CRM (mock)"):
            payload_preview = top_leads[["Tak-ID", "Adress", "Lead Score", "Besparing (kSEK/år)"]].to_dict(orient="records")
            st.success("Skickade 5 topp-leads (mock).")
            with st.expander("Payload (mock)"):
                st.json(payload_preview)

    # Metod & antaganden
    with st.expander("Metod & antaganden (mock)"):
        st.markdown(
            f"""
- **PV-potential:** kWp ≈ area / {M2_PER_KWP} m²/ kWp. kWh/år ≈ kWp × {SE_LATLON_IRRADIANCE_KWH_PER_KWP}.
- **Besparing:** kWh × elpris (kr/kWh) → visas i kSEK/år.
- **Lead Score (0–100):** viktar area, inkomst, elbil, skuggindex, takålder och estimerad besparing.
- **Skuggindex:** 0 (ingen skugga) – 1 (mycket skugga). Här slumpas värdet tills bildanalys kopplas på.
- **Karta:** punkter genereras runt centrum för vald ort/postnr (mockade koordinater).
"""
        )

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    dashboard()
