"""Hotel booking cancellation predictor — a small demo UI for the saved models.

Run from the project root:
    .venv\\Scripts\\streamlit run app\\app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
RESULTS = ROOT / 'reports' / 'results'
DATA = ROOT / 'data' / 'processed'

# model name -> (saved file, results file that holds its threshold)
MODEL_FILES = {
    'XGBoost (final model)': ('xgboost', 'xgboost'),
    'Random Forest': ('random_forest', 'random_forest'),
    'Neural Network': ('neural_network', 'neural_network'),
    'Decision Tree': ('decision_tree', 'decision_tree'),
    'Logistic Regression': ('logistic_regression', 'logistic_regression'),
    'Ensemble (RF + XGB + NN)': ('ensemble', 'ensemble'),
}

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']

# column order the models were trained on (same as train.csv without is_canceled)
FEATURES = ['hotel', 'lead_time', 'arrival_date_month', 'arrival_date_day_of_month',
            'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
            'meal', 'country', 'market_segment', 'distribution_channel', 'is_repeated_guest',
            'previous_cancellations', 'previous_bookings_not_canceled', 'reserved_room_type',
            'deposit_type', 'agent', 'customer_type', 'adr', 'total_of_special_requests',
            'total_nights', 'total_guests', 'is_family']

# starting values of the form (a typical booking)
DEFAULTS = dict(
    hotel='City Hotel', lead_time=50, arrival_date_month='July', arrival_date_day_of_month=15,
    stays_in_weekend_nights=1, stays_in_week_nights=2, adults=2, children=0, babies=0,
    meal='BB', country='PRT', market_segment='Online TA', distribution_channel='TA/TO',
    is_repeated_guest=0, previous_cancellations=0, previous_bookings_not_canceled=0,
    reserved_room_type='A', deposit_type='No Deposit', agent='9', customer_type='Transient',
    adr=100.0, total_of_special_requests=0)

# two ready-made bookings for the demo
EXAMPLES = {
    'High-risk example': dict(
        hotel='City Hotel', lead_time=250, arrival_date_month='June', arrival_date_day_of_month=15,
        stays_in_weekend_nights=1, stays_in_week_nights=2, adults=2, children=0, babies=0,
        meal='BB', country='PRT', market_segment='Online TA', distribution_channel='TA/TO',
        is_repeated_guest=0, previous_cancellations=1, previous_bookings_not_canceled=0,
        reserved_room_type='A', deposit_type='No Deposit', agent='9', customer_type='Transient',
        adr=110.0, total_of_special_requests=0),
    'Low-risk example': dict(
        hotel='Resort Hotel', lead_time=7, arrival_date_month='August', arrival_date_day_of_month=10,
        stays_in_weekend_nights=2, stays_in_week_nights=5, adults=2, children=1, babies=0,
        meal='HB', country='GBR', market_segment='Direct', distribution_channel='Direct',
        is_repeated_guest=1, previous_cancellations=0, previous_bookings_not_canceled=2,
        reserved_room_type='D', deposit_type='No Deposit', agent='0', customer_type='Transient',
        adr=150.0, total_of_special_requests=2),
}


@st.cache_resource
def load_model(slug):
    return joblib.load(MODELS / f'{slug}.joblib')


@st.cache_data
def load_threshold(slug):
    return float(pd.read_csv(RESULTS / f'{slug}.csv')['threshold'].iloc[0])


@st.cache_data
def load_train():
    # used only to fill the dropdown lists with the values the models have seen
    return pd.read_csv(DATA / 'train.csv', dtype={'agent': str})


@st.cache_data
def load_test():
    return pd.read_csv(DATA / 'test.csv', dtype={'agent': str})


def options(train, col):
    """Dropdown values, most common first."""
    return train[col].value_counts().index.tolist()


def add_derived(row):
    """The three features built in notebook 02, computed from the form fields."""
    row['total_nights'] = row['stays_in_weekend_nights'] + row['stays_in_week_nights']
    row['total_guests'] = row['adults'] + row['children'] + row['babies']
    row['is_family'] = int(row['children'] > 0 or row['babies'] > 0)
    return row


def show_result(proba, threshold):
    will_cancel = proba >= threshold
    c1, c2 = st.columns(2)
    c1.metric('Cancellation probability', f'{proba:.1%}')
    c2.metric('Prediction', 'Likely to CANCEL' if will_cancel else 'Likely to STAY')
    st.progress(min(max(proba, 0.0), 1.0))
    if will_cancel:
        st.error(f'Probability {proba:.1%} is above the threshold {threshold:.1%}. '
                 'Suggested action: send a reminder, ask for a deposit, or plan for this room '
                 'possibly becoming free.')
    else:
        st.success(f'Probability {proba:.1%} is below the threshold {threshold:.1%}. '
                   'No action needed for this booking.')


# ---------------------------------------------------------------- page
st.set_page_config(page_title='Hotel Cancellation Predictor', page_icon='🏨', layout='wide')
st.title('🏨 Hotel Booking Cancellation Predictor')
st.caption('IT3091 Machine Learning · Group 2026-AI-46 · enter a booking and the saved model '
           'predicts whether it will be cancelled.')

train = load_train()

with st.sidebar:
    st.header('Model')
    model_name = st.selectbox('Choose a saved model', list(MODEL_FILES))
    slug, result_slug = MODEL_FILES[model_name]
    model = load_model(slug)
    threshold = load_threshold(result_slug)
    st.write(f'File: `models/{slug}.joblib`')
    st.write(f'Decision threshold: **{threshold:.3f}**')
    st.caption('A booking is flagged as "will cancel" when its probability is at or above the '
               'threshold. Each threshold was chosen in that model\'s notebook (best F1 on '
               'training data).')

tab_form, tab_check = st.tabs(['📝 Predict one booking', '✅ Check on real test bookings'])

# ---------------------------------------------------------------- tab 1: form
with tab_form:
    st.write('Fill in the booking, or load an example first.')
    # the form reads its values from session_state, so the example buttons can overwrite them
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    b1, b2, _ = st.columns([1, 1, 3])
    for button_col, name in zip([b1, b2], EXAMPLES):
        if button_col.button(name):
            for key, value in EXAMPLES[name].items():
                st.session_state[key] = value

    agent_options = options(train, 'agent')

    with st.form('booking'):
        st.subheader('Booking')
        c1, c2, c3, c4 = st.columns(4)
        hotel = c1.selectbox('Hotel', options(train, 'hotel'), key='hotel')
        lead_time = c2.number_input('Lead time (days before arrival)', 0, 800, key='lead_time')
        month = c3.selectbox('Arrival month', MONTHS, key='arrival_date_month')
        day = c4.number_input('Arrival day of month', 1, 31, key='arrival_date_day_of_month')

        c1, c2, c3, c4 = st.columns(4)
        market = c1.selectbox('Market segment', options(train, 'market_segment'), key='market_segment')
        channel = c2.selectbox('Distribution channel', options(train, 'distribution_channel'),
                               key='distribution_channel')
        customer = c3.selectbox('Customer type', options(train, 'customer_type'), key='customer_type')
        deposit = c4.selectbox('Deposit type', options(train, 'deposit_type'), key='deposit_type')

        c1, c2 = st.columns(2)
        agent = c1.selectbox('Travel agent ID ("0" = no agent)', agent_options, key='agent')
        adr = c2.number_input('Average daily rate (€)', 0.0, 600.0, step=5.0, key='adr')

        st.subheader('Stay')
        c1, c2, c3, c4, c5 = st.columns(5)
        weekend = c1.number_input('Weekend nights', 0, 20, key='stays_in_weekend_nights')
        week = c2.number_input('Week nights', 0, 50, key='stays_in_week_nights')
        meal = c3.selectbox('Meal plan', options(train, 'meal'), key='meal')
        room = c4.selectbox('Reserved room type', sorted(options(train, 'reserved_room_type')),
                            key='reserved_room_type')
        requests = c5.number_input('Special requests', 0, 5, key='total_of_special_requests')

        st.subheader('Guests')
        c1, c2, c3, c4 = st.columns(4)
        adults = c1.number_input('Adults', 0, 10, key='adults')
        children = c2.number_input('Children', 0, 10, key='children')
        babies = c3.number_input('Babies', 0, 10, key='babies')
        country = c4.selectbox('Country (ISO code)', options(train, 'country'), key='country')

        c1, c2, c3 = st.columns(3)
        repeated = c1.selectbox('Repeated guest?', [0, 1], format_func=lambda v: 'Yes' if v else 'No',
                                key='is_repeated_guest')
        prev_cancel = c2.number_input('Previous cancellations', 0, 30, key='previous_cancellations')
        prev_ok = c3.number_input('Previous bookings not cancelled', 0, 80,
                                  key='previous_bookings_not_canceled')

        submitted = st.form_submit_button('Predict', type='primary')

    if submitted:
        if adults + children + babies == 0:
            st.warning('A booking needs at least one guest.')
        else:
            row = add_derived({
                'hotel': hotel, 'lead_time': lead_time, 'arrival_date_month': month,
                'arrival_date_day_of_month': day, 'stays_in_weekend_nights': weekend,
                'stays_in_week_nights': week, 'adults': adults, 'children': children,
                'babies': babies, 'meal': meal, 'country': country, 'market_segment': market,
                'distribution_channel': channel, 'is_repeated_guest': repeated,
                'previous_cancellations': prev_cancel, 'previous_bookings_not_canceled': prev_ok,
                'reserved_room_type': room, 'deposit_type': deposit, 'agent': agent,
                'customer_type': customer, 'adr': adr, 'total_of_special_requests': requests,
            })
            booking = pd.DataFrame([row])[FEATURES]
            proba = float(model.predict_proba(booking)[:, 1][0])

            st.subheader(f'Result — {model_name}')
            show_result(proba, threshold)
            with st.expander('What was sent to the model'):
                st.dataframe(booking.T.astype(str).rename(columns={0: 'value'}), use_container_width=True)

# ---------------------------------------------------------------- tab 2: real test bookings
with tab_check:
    st.write('Pick random bookings from the **test set** (bookings the model never saw during '
             'training), predict them, and compare with what really happened.')
    test = load_test()
    c1, c2 = st.columns([1, 3])
    n = c1.number_input('How many bookings?', 5, 500, 20, step=5)
    seed = c2.number_input('Random seed (change it to draw other bookings)', 0, 10_000, 42)

    if st.button('Draw and predict', type='primary'):
        sample = test.sample(int(n), random_state=int(seed))
        proba = model.predict_proba(sample[FEATURES])[:, 1]
        table = pd.DataFrame({
            'hotel': sample['hotel'].values,
            'lead_time': sample['lead_time'].values,
            'market_segment': sample['market_segment'].values,
            'probability': [f'{p:.1%}' for p in proba],
            'predicted': ['cancel' if p >= threshold else 'stay' for p in proba],
            'actual': ['cancel' if y == 1 else 'stay' for y in sample['is_canceled']],
        })
        is_right = table['predicted'] == table['actual']
        table['correct'] = ['✓' if ok else '✗' for ok in is_right]

        correct = int(is_right.sum())
        c1, c2, c3 = st.columns(3)
        c1.metric('Correct predictions', f'{correct} / {len(table)}')
        c2.metric('Accuracy on this sample', f'{correct / len(table):.0%}')
        actual_cancels = (table['actual'] == 'cancel').sum()
        caught = ((table['actual'] == 'cancel') & (table['predicted'] == 'cancel')).sum()
        c3.metric('Cancellations caught', f'{caught} / {actual_cancels}')

        st.dataframe(table.style.map(
            lambda ok: 'background-color: #2e7d32' if ok == '✓' else 'background-color: #c62828',
            subset=['correct']), use_container_width=True, hide_index=True)
        st.caption('Green = the model was right, red = it was wrong. On the full test set '
                   '(16,994 bookings) XGBoost is 81.9% accurate and catches 76.9% of cancellations.')
