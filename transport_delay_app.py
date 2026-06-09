#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This app is designed to run as an interactive CLI.
# To launch it, run the following in your terminal:
#
#   python transport_delay_app.py
#
# Make sure your .env file is set up with your API keys first.


# <h4>SECTION 7: INTERACTIVE CLI RECOMMENDATION APP</h4>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
import datetime
import shap
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# In[ ]:


# ── 7.1  API Keys ─────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
TICKETMASTER_API_KEY = "YOUR_TICKETMASTER_API_KEY"

CITY = "Toronto"
COUNTRY_CODE = "CA"


# In[ ]:


# ── 7.2  Load Data & Rebuild Models ────────────────────────────────────────────
def load_models():
    df = pd.read_csv('transport_features.csv')
 
    DROP_COLS = [
        'trip_id', 'date', 'time', 'scheduled_departure', 'scheduled_arrival',
        'weather_condition', 'event_type', 'event_label', 'season',
        'transport_type', 'route_id', 'origin_station', 'destination_station',
    ]
    TARGET_REG = 'actual_arrival_delay_min'
    TARGET_CLF = 'delayed'
 
    feature_cols = [c for c in df.columns if c not in DROP_COLS +
                    [TARGET_REG, 'actual_departure_delay_min', TARGET_CLF]]
 
    X = df[feature_cols]
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]
 
    X_train, X_temp, y_reg_train, y_reg_temp, y_clf_train, y_clf_temp = train_test_split(
        X, y_reg, y_clf, test_size=0.30, random_state=42, stratify=y_clf)
    X_val, X_test, y_reg_val, y_reg_test, y_clf_val, y_clf_test = train_test_split(
        X_temp, y_reg_temp, y_clf_temp, test_size=0.50, random_state=42, stratify=y_clf_temp)
 
    def assign_risk(d):
        return 0 if d < 10 else (1 if d < 20 else 2)
 
    y_risk_train = y_reg_train.apply(assign_risk)
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_risk_train)
 
    lr_reg = LinearRegression().fit(X_train, y_reg_train)
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1).fit(X_sm, y_sm)
 
    explainer_reg = shap.LinearExplainer(lr_reg, X_train)
 
    # Lookup tables
    route_list = sorted(df['route_id'].unique())
    station_list = sorted(pd.concat([df['origin_station'], df['destination_station']]).unique())
    transport_types = ['Bus', 'Metro', 'Train', 'Tram']
    transport_enc = {t: i for i, t in enumerate(transport_types)}
    route_enc = {r: i for i, r in enumerate(route_list)}
    station_enc = {s: i for i, s in enumerate(station_list)}
    route_avg_delay = df.groupby('route_id')['actual_arrival_delay_min'].mean().to_dict()
    transport_avg = df.groupby('transport_type')['actual_arrival_delay_min'].mean().to_dict()
 
    return (
        lr_reg, 
        rf_clf, 
        explainer_reg, 
        feature_cols,
        X_train, 
        y_reg_train,
        route_list, 
        station_list, 
        transport_types,
        transport_enc, 
        route_enc, 
        station_enc,
        route_avg_delay, 
        transport_avg, 
        df
        )


# In[ ]:


# ── 7.3  API Functions ─────────────────────────────────────────────────────────
def fetch_weather():
    """Retrieve current weather conditions from OpenWeatherMap API"""
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
        return None
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={'q': f"{CITY},{COUNTRY_CODE}",
                    'appid': OPENWEATHER_API_KEY, 'units': 'metric'},
            timeout=10
            )

        r.raise_for_status()
        d = r.json()

        # Convert weather condition ID into simplified categories used by the model
        wid = d['weather'][0]['id']
        if   wid < 300  : cond = 'Storm'
        elif wid < 600  : cond = 'Rain'
        elif wid < 700  : cond = 'Snow'
        elif wid == 741 : cond = 'Fog'
        elif wid > 800  : cond = 'Cloudy'
        else:             cond = 'Clear'

        return {
            'temperature_C'    : d['main']['temp'],
            'humidity_percent' : d['main']['humidity'],
            'wind_speed_kmh'   : d['wind']['speed'] * 3.6,
            'precipitation_mm' : d.get('rain', {}).get('1h', 0.0),
            'weather_condition': cond,
        }

    except Exception:
        return None


def fetch_events():
    """Retrieve today's major events from the Ticketmaster API"""
    if TICKETMASTER_API_KEY == "YOUR_TICKETMASTER_API_KEY":
        return None
    try:
        today = datetime.date.today()
        r = requests.get(
            "https://app.ticketmaster.com/discovery/v2/events.json",
            params={'apikey': TICKETMASTER_API_KEY, 'city': CITY,
                    'countryCode': COUNTRY_CODE,
                    'startDateTime': f"{today}T00:00:00Z",
                    'endDateTime'  : f"{today}T23:59:59Z",
                    'size': 10, 'sort': 'relevance,desc'},
            timeout=10
            )

        r.raise_for_status()
        events = r.json().get('_embedded', {}).get('events', [])

        if not events:
            return {'event_type': 'No Event', 'event_attendance_est': 0,
                    'event_name': None}

        ev = events[0]
        seg = (ev.get('classifications', [{}])[0]
               .get('segment', {}).get('name', '').lower())

        # Categorize event type into Sports, Concert, or Festival
        etype = ('Sports' if 'sport' in seg else
                 'Concert' if 'music' in seg else 'Festival')
        
        # Estimate attendance using venue capacity when available
        cap = None
        if ev.get('_embedded'):
            cap = ev.get('venue', {}).get('capacity')

        return {
            'event_type'          : etype,
            'event_attendance_est': int(cap) if cap else 5000,
            'event_name'          : ev.get('name', 'Unknown event'),
        }

    except Exception:
        return None


# In[ ]:


# ── 7.4  Feature Builder ─────────────────────────────────────────────────────────
# Map event categories to severity scores used in feature engineering
event_severity_map = {
    'No Event': 0, 'Concert': 1, 'Festival': 2,
    'Sports'  : 3, 'Parade' : 4, 'Protest' : 5,
}
 
def build_feature_vector(
    arrival_hour, transport_type, route_id,
    origin_station, destination_station,
    weather, events, scheduled_duration_min,
    holiday, traffic_congestion_index,
    feature_cols, transport_enc, route_enc, station_enc
):
    """Construct a single feature vector from user inputs, weather, and event data"""

    # Extract current calendar information
    now = datetime.datetime.now()
    month = now.month
    weekday = now.weekday()
    
    # Create rush-hour indicators
    is_morning_rush = int(7 <= arrival_hour <= 9)
    is_evening_rush = int(16 <= arrival_hour <= 19)

    # Identify whether the trip occurs during a peak commuting period
    peak_hour = int(is_morning_rush or is_evening_rush)

    # Create weekend indicator
    is_weekend = int(weekday >= 5)
    
    # Categorize arrival time into overnight, morning, afternoon, or evening
    if   arrival_hour < 6 : time_of_day = 0
    elif arrival_hour < 12: time_of_day = 1
    elif arrival_hour < 17: time_of_day = 2
    else:                   time_of_day = 3
    
    # Encode cyclic hour information using sine/cosine transformation
    hour_sin = np.sin(2 * np.pi * arrival_hour / 24)
    hour_cos = np.cos(2 * np.pi * arrival_hour / 24)
    
    # Generate seasonal indicators (Autumn is the reference category)
    m = month
    season_Spring = int(m in [3,4,5])
    season_Summer = int(m in [6,7,8])
    season_Winter = int(m in [12,1,2])
    
    # Create weather severity features
    wc = weather['weather_condition']
    is_bad_weather = int(wc in ['Rain','Snow','Storm'])
    
    # Convert event information into model-friendly numerical features
    event_type = events.get('event_type', 'No Event')
    event_severity = event_severity_map.get(event_type, 0)
    has_event = int(event_severity > 0)
    event_attendance = events.get('event_attendance_est', 0)
 
    row = {
        'temperature_C'           : weather['temperature_C'],
        'humidity_percent'        : weather['humidity_percent'],
        'wind_speed_kmh'          : weather['wind_speed_kmh'],
        'precipitation_mm'        : weather['precipitation_mm'],
        'event_attendance_est'    : event_attendance,
        'traffic_congestion_index': traffic_congestion_index,
        'holiday'                 : holiday,
        'peak_hour'               : peak_hour,
        'weekday'                 : weekday,
        'hour'                    : arrival_hour,
        'month'                   : month,
        'is_morning_rush'         : is_morning_rush,
        'is_evening_rush'         : is_evening_rush,
        'time_of_day'             : time_of_day,
        'scheduled_duration_min'  : scheduled_duration_min,
        'hour_sin'                : hour_sin,
        'hour_cos'                : hour_cos,
        'weather_Cloudy'          : int(wc == 'Cloudy'),
        'weather_Fog'             : int(wc == 'Fog'),
        'weather_Rain'            : int(wc == 'Rain'),
        'weather_Snow'            : int(wc == 'Snow'),
        'weather_Storm'           : int(wc == 'Storm'),
        'is_bad_weather'          : is_bad_weather,
        'event_severity'          : event_severity,
        'has_event'               : has_event,
        'season_Spring'           : season_Spring,
        'season_Summer'           : season_Summer,
        'season_Winter'           : season_Winter,
        'transport_type_enc'      : transport_enc.get(transport_type, 0),           # Numerical encoding of transport mode
        'route_id_enc'            : route_enc.get(route_id, 0),                     # Numerical encoding of route
        'origin_station_enc'      : station_enc.get(origin_station, 0),             # Numerical encoding of origin station
        'destination_station_enc' : station_enc.get(destination_station, 0),        # Numerical encoding of destination station
        'bad_weather_x_peak'      : is_bad_weather * peak_hour,                     # Interaction feature: impact of bad weather during peak periods
        'event_x_attendance'      : event_severity * np.log1p(event_attendance),    # Interaction feature: combines event severity and attendance size
        'congestion_x_peak'       : traffic_congestion_index * peak_hour,           # Interaction feature: congestion amplified during rush hours
        'hour_x_bad_weather'      : arrival_hour * is_bad_weather,                  # Interaction feature: weather impact varies by time of day
        'is_weekend'              : is_weekend,                                     
        'holiday_x_weekend'       : holiday * is_weekend,                           # Interaction feature: holidays occurring on weekends
    }

    return pd.DataFrame([row])[feature_cols]


# In[ ]:


# ── 7.5  Prediction ─────────────────────────────────────────────────────────────
risk_labels = {0: 'Low 🟢', 1: 'Medium 🟡', 2: 'High 🔴'}
 
def run_prediction(fv, lr_reg, rf_clf, explainer_reg,
                   feature_cols, transport_avg,
                   arrival_time_str, scheduled_duration_min):
    """Generate delay prediction, risk classification, explanations, and recommendations"""

    # Predict expected arrival delay (cannot be negative)
    predicted_delay = max(float(lr_reg.predict(fv)[0]), 0)
    risk_class = int(rf_clf.predict(fv)[0])
    risk_proba = rf_clf.predict_proba(fv)[0]
    
    # Calculate recommended departure time based on predicted delay
    arrival_dt = datetime.datetime.strptime(arrival_time_str, "%H:%M")
    departure_dt = arrival_dt - datetime.timedelta(
        minutes=scheduled_duration_min + predicted_delay)
    recommended_dep = departure_dt.strftime("%H:%M")
    
    # Compute SHAP values for this specific trip
    shap_vals = explainer_reg.shap_values(fv)[0].astype(float)
    shap_series = pd.Series(shap_vals, index=feature_cols)
    top_drivers = shap_series.nlargest(3)       # Identify top factors increasing predicted delay
    top_reducers = shap_series.nsmallest(3)     # Identify top factors reducing predicted delay
 
    # High-delay scenario: recommend leaving earlier
    if predicted_delay > 20:
        biz_icon = "🔴"
        biz_msg  = (f"High delay expected. "
                    f"Leave by {departure_dt.strftime('%H:%M')} "
                    f"instead of your usual time.")

    # Medium-delay scenario: suggest caution and alternative routes
    elif predicted_delay > 10:
        biz_icon = "🟡"
        biz_msg  = (f"Moderate delay expected. "
                    f"Consider leaving a few minutes earlier "
                    f"or using a faster route.")
    
    # Low-delay scenario: current travel plan appears acceptable
    else:
        biz_icon = "🟢"
        biz_msg  = (f"Low delay expected. "
                    f"Your current route and time look good.")
 
    return {
        'predicted_delay'      : round(predicted_delay, 1),
        'risk_class'           : risk_class,
        'risk_label'           : risk_labels[risk_class],
        'risk_proba'           : risk_proba,
        'recommended_departure': recommended_dep,
        'transport_avg'        : transport_avg,
        'top_drivers'          : top_drivers,
        'top_reducers'         : top_reducers,
        'shap_series'          : shap_series,
        'biz_icon'             : biz_icon,
        'biz_msg'              : biz_msg,
    }


# In[ ]:


# ── 7.6  Output Display ──────────────────────────────────────────────────────
def display_result(inputs, weather, events, result, explainer_reg):
 
    SEP = "─" * 55
 
    print("\n")
    print("  🚌  TRIP RECOMMENDATION RESULT")
    print(SEP)
    print(f"  Destination   : {inputs['destination']}")
    print(f"  Arrival time  : {inputs['arrival_time']}")
    print(f"  Transport     : {inputs['transport_type']}")
    print(f"  Route         : {inputs['route_id']}")
    print(SEP)
    print(f"  🌤  Weather   : {weather['weather_condition']}, "
          f"{weather['temperature_C']:.1f}°C, "
          f"Humidity {weather['humidity_percent']:.0f}%, "
          f"Wind {weather['wind_speed_kmh']:.0f} km/h")
    ev_str = (f"{events['event_type']} — {events['event_name']}"
              if events.get('event_name') else "No events today")
    print(f"  🎟  Events    : {ev_str}")
    print(SEP)
    print(f"  Predicted Delay        : {result['predicted_delay']} min")
    print(f"  Commute Risk           : {result['risk_label']}")
    print(f"  Risk probability       : "
          f"Low {result['risk_proba'][0]:.0%}  "
          f"Med {result['risk_proba'][1]:.0%}  "
          f"High {result['risk_proba'][2]:.0%}")
    print(f"  ⏰ Recommended departure : {result['recommended_departure']}")
    print(SEP)
    print("  📊 Route Comparison (avg historical delay):")
    for mode, avg in result['transport_avg'].items():
        marker = "  ◄ your choice" if mode == inputs['transport_type'] else ""
        print(f"      {mode:8s}: {avg:.1f} min{marker}")
    print(SEP)
    print("  🔍 Why is this predicted?")
    print("     Delay drivers (+):")
    for feat, val in result['top_drivers'].items():
        print(f"       {feat:32s}: +{val:.2f} min")
    print("     Delay reducers (-):")
    for feat, val in result['top_reducers'].items():
        print(f"       {feat:32s}: {val:.2f} min")
    print(SEP)
    print("  💡 RECOMMENDATION")
    # Word wrap the recommendation message
    words = f"  {result['biz_icon']}  {result['biz_msg']}".split()
    line  = "  "
    for word in words:
        if len(line) + len(word) + 1 > 55:
            print(f"  {line}")
            line = "  " + word
        else:
            line += (" " if line.strip() else "") + word
    if line.strip():
        print(f"  {line}")
    print(SEP)


# In[ ]:


def display_chart(inputs, weather, events, result, explainer_reg, feature_cols):
    """Show a compact visual summary for the predicted trip."""
 
    DARK='#0f1117'; CARD='#1a1d27'; ACCENT='#4f8ef7'; ACCENT2='#f7c34f'
    ACCENT3='#f7614f'; ACCENT4='#4ff7b8'; TEXT='#e8eaf0'; SUBTEXT='#8b90a0'
    RISK_COLORS = {0: ACCENT4, 1: ACCENT2, 2: ACCENT3}
 
    plt.rcParams.update({
        'figure.facecolor':DARK, 'axes.facecolor':CARD,
        'axes.edgecolor':'#2a2d3a', 'axes.labelcolor':TEXT,
        'xtick.color':SUBTEXT, 'ytick.color':SUBTEXT,
        'text.color':TEXT, 'grid.color':'#2a2d3a',
        'font.family':'DejaVu Sans', 'axes.titlesize':10, 'axes.labelsize':8,
    })
 
    fig = plt.figure(figsize=(16, 6), facecolor=DARK)
    fig.suptitle(
        f"Trip to {inputs['destination']}  |  "
        f"Arr {inputs['arrival_time']}  |  "
        f"{weather['weather_condition']} {weather['temperature_C']:.1f}°C  |  "
        f"Risk: {result['risk_label']}  |  "
        f"Leave by: {result['recommended_departure']}",
        fontsize=11, color=TEXT, fontweight='bold', y=1.01)
 
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                           left=0.06, right=0.97, top=0.88, bottom=0.12)


    # ── Plot 1 · Risk probability gauge ──────────────────────────────────────
    # Visualize the model's confidence across Low, Medium, and High delay-risk classes
    ax1 = fig.add_subplot(gs[0, 0])
    proba = result['risk_proba']

    bars = ax1.barh(['High','Medium','Low'], [proba[2],proba[1],proba[0]],
                    color=[ACCENT3, ACCENT2, ACCENT4],
                    edgecolor='none', height=0.5)
    for bar, val in zip(bars, [proba[2],proba[1],proba[0]]):
        ax1.text(val + 0.02, bar.get_y()+bar.get_height()/2,
                 f'{val:.0%}', va='center', color=TEXT, fontsize=9)
    ax1.set_xlim(0, 1.15)
    ax1.set_title(f'Commute Risk\n{result["risk_label"]}')
    ax1.grid(axis='x', alpha=0.3)
 

    # ── Plot 2 · Route comparison ─────────────────────────────────────────────
    # Compare historical average delays across transport modes to help users evaluate alternative travel options.
    ax2 = fig.add_subplot(gs[0, 1])
    modes = list(result['transport_avg'].keys())
    delays = list(result['transport_avg'].values())

    colors = [ACCENT3 if m == inputs['transport_type'] else ACCENT for m in modes]
    b2 = ax2.bar(modes, delays, color=colors, edgecolor='none', width=0.5)
    for bar, val in zip(b2, delays):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f'{val:.1f}', ha='center', va='bottom', color=TEXT, fontsize=8)
    ax2.set_title('Route Comparison\n(average historical delay)')
    ax2.set_ylabel('Average Delay (min)')
    ax2.set_ylim(0, max(delays)*1.25); ax2.grid(axis='y', alpha=0.3)
 

    # ── Plot 3 · SHAP waterfall ───────────────────────────────────────────────
    # Explain how the most influential features combine to move the prediction from the baseline delay to the final predicted delay.
    ax3 = fig.add_subplot(gs[0, 2])
    sv = result['shap_series']
    top8 = sv.abs().nlargest(8).index
    sv_top = sv[top8]
    base = float(explainer_reg.expected_value)
    cum = base

    positions = []
    for val in sv_top.values:
        positions.append(cum); cum += val
        
    colors_wf = [ACCENT4 if v > 0 else ACCENT3 for v in sv_top.values]
    ax3.barh(range(len(sv_top)), sv_top.values.astype(float),
             left=[float(p) for p in positions],
             color=colors_wf, edgecolor='none', height=0.6)
    ax3.axvline(base, color=SUBTEXT, lw=0.8, ls='--',
                label=f'Base={base:.1f}')
    ax3.axvline(base + float(sv_top.sum()),
                color=ACCENT2, lw=1.5,
                label=f"Pred={result['predicted_delay']} min")
    ax3.set_yticks(range(len(sv_top)))
    ax3.set_yticklabels(sv_top.index, fontsize=7)
    ax3.set_title('SHAP Explanation\n(top 8 feature contributions)')
    ax3.set_xlabel('Delay contribution (min)')
    ax3.legend(fontsize=7.5, framealpha=0.2); ax3.grid(axis='x', alpha=0.3)
 
    plt.tight_layout()
    plt.show()


# In[ ]:


# ── 7.7  Interactive Input Loop ─────────────────────────────────────────────
def get_input(prompt, valid_options=None, input_type=str,
              default=None, min_val=None, max_val=None):
    """Helper — prompt user, validate, return clean value"""
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"  {prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            value = input_type(raw)

            # Validate against allowed options
            if valid_options and value not in valid_options:
                print(f"    ⚠ Please choose from: {valid_options}")
                continue
            
            # Validate minimum value constraint
            if min_val is not None and value < min_val:
                print(f"    ⚠ Minimum value is {min_val}")
                continue
            
            # Validate maximum value constraint
            if max_val is not None and value > max_val:
                print(f"    ⚠ Maximum value is {max_val}")
                continue

            return value

        except (ValueError, TypeError):
            print(f"    ⚠ Invalid input, please try again")
 
 
def run_app():
    # ── Startup ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("   🚌  PUBLIC TRANSPORT DELAY PREDICTOR")
    print("   Personalized commute recommendations")
    print("═" * 55)
 
    (lr_reg, rf_clf, explainer_reg, feature_cols,
     X_train, y_reg_train,
     route_list, station_list, transport_types,
     transport_enc, route_enc, station_enc,
     route_avg_delay, transport_avg, df) = load_models()
 
    # ── Fetch live conditions once ────────────────────────────────────────────
    print(f"\n  Fetching live conditions for {CITY} …", end="", flush=True)
    weather = fetch_weather()
    events = fetch_events()
 
    if weather is None:
        print(" ⚠ API unavailable — using demo weather data")
        weather = {
            'temperature_C'    : 8.0,
            'humidity_percent' : 72.0,
            'wind_speed_kmh'   : 20.0,
            'precipitation_mm' : 2.5,
            'weather_condition': 'Rain',
        }
    else:
        print(" ✓")
 
    if events is None:
        print("  Events API unavailable — assuming no events today")
        events = {'event_type': 'No Event',
                  'event_attendance_est': 0, 'event_name': None}
 

    print(f"\n  Current conditions in {CITY}:")
    print(f"    🌤  {weather['weather_condition']}  |  "
          f"{weather['temperature_C']:.1f}°C  |  "
          f"Humidity {weather['humidity_percent']:.0f}%  |  "
          f"Wind {weather['wind_speed_kmh']:.0f} km/h  |  "
          f"Rain {weather['precipitation_mm']:.1f} mm")

    ev_display = (f"{events['event_type']} — {events['event_name']}"
                  if events.get('event_name') else "No events detected")
    print(f"    🎟  {ev_display}")
 
    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        print("\n" + "─" * 55)
        print("  📍 TRIP DETAILS")
        print("─" * 55)
 
        # 1. Destination
        destination = input("  Enter your destination: ").strip()
        if not destination:
            destination = "My Destination"
 
        # 2. Arrival time
        while True:
            arrival_time = input("  Desired arrival time (HH:MM, 24h): ").strip()
            try:
                datetime.datetime.strptime(arrival_time, "%H:%M")
                break
            except ValueError:
                print("    ⚠ Please use HH:MM format, e.g. 09:00 or 17:30")
 
        arrival_hour = int(arrival_time.split(':')[0])
 
        # 3. Transport type
        print("\n  Transport type:")
        for i, t in enumerate(transport_types, 1):
            print(f"    {i}. {t}")
        t_idx = get_input("  Your choice (1–4)", input_type=int,
                          valid_options=[1,2,3,4])
        transport_type = transport_types[t_idx - 1]
 
        # 4. Route — filtered to selected transport type (top 10)
        transport_routes = sorted(
            df[df['transport_type'] == transport_type]['route_id'].unique()
        )
        display_routes = transport_routes[:10]
        print(f"\n  Available routes for {transport_type} (top 10):")
        for i, r in enumerate(display_routes, 1):
            avg = route_avg_delay.get(r, 0)
            print(f"    {i:>2}. {r:<12s} (avg delay: {avg:.1f} min)")
        print(f"    (Press Enter to use {display_routes[0]} as default)")
        r_idx = get_input(f"  Your choice (1–{len(display_routes)})",
                        input_type=int,
                        valid_options=list(range(1, len(display_routes)+1)),
                        default=1)
        route_id = display_routes[r_idx - 1]
 
        # 5. Origin station
        print(f"\n  Origin station (1–{len(station_list)}, e.g. 1 = {station_list[0]}):")
        o_idx = get_input("  Origin station number",
                          input_type=int,
                          valid_options=list(range(1, len(station_list)+1)),
                          default=1)
        origin_station = station_list[o_idx - 1]
 
        # 6. Destination station
        d_idx = get_input(f"  Destination station number (1–{len(station_list)})",
                          input_type=int,
                          valid_options=list(range(1, len(station_list)+1)),
                          default=10)
        destination_station = station_list[d_idx - 1]
 
        # 7. Scheduled travel duration
        duration = get_input("  Scheduled travel duration (minutes)",
                             input_type=float, default=30.0,
                             min_val=1.0, max_val=180.0)
 
        # 8. Traffic congestion
        congestion = get_input("  Traffic congestion level (1–100, 50=average)",
                               input_type=float, default=50.0,
                               min_val=1.0, max_val=100.0)
 
        # 9. Holiday
        holiday_str = get_input("  Is today a public holiday? (y/n)",
                                valid_options=['y','n','Y','N'], default='n')
        holiday = 1 if holiday_str.lower() == 'y' else 0
 
        # ── Build & predict ───────────────────────────────────────────────────
        print("\n  ⚙  Running prediction …", end="", flush=True)
 
        fv = build_feature_vector(
            arrival_hour=arrival_hour,
            transport_type=transport_type,
            route_id=route_id,
            origin_station=origin_station,
            destination_station=destination_station,
            weather=weather,
            events=events,
            scheduled_duration_min=duration,
            holiday=holiday,
            traffic_congestion_index=congestion,
            feature_cols=feature_cols,
            transport_enc=transport_enc,
            route_enc=route_enc,
            station_enc=station_enc,
        )
 
        result = run_prediction(
            fv, lr_reg, rf_clf, explainer_reg,
            feature_cols, transport_avg,
            arrival_time, duration,
        )
        print(" ✓")
 
        inputs = {
            'destination'   : destination,
            'arrival_time'  : arrival_time,
            'transport_type': transport_type,
            'route_id'      : route_id,
        }
 
        # ── Display results ───────────────────────────────────────────────────
        display_result(inputs, weather, events, result, explainer_reg)
 
        # ── Show chart ───────────────────────────────────────────────────────
        show_chart = get_input("\n  Show visual chart? (y/n)",
                               valid_options=['y','n','Y','N'], default='y')
        if show_chart.lower() == 'y':
            display_chart(inputs, weather, events, result,
                          explainer_reg, feature_cols)
 
        # ── Another trip ─────────────────────────────────────────────────────
        again = get_input("\n  Predict another trip? (y/n)",
                          valid_options=['y','n','Y','N'], default='n')
        if again.lower() != 'y':
            print("\n  Thanks for using the Public Transport Delay Predictor!")
            print("  Safe travels 🚌\n")
            break


# In[ ]:


# ── 7.8  Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_app()

