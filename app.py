import streamlit as st
import logging
import snowflake.connector
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import calendar
from streamlit_option_menu import option_menu

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def init_snowflake_connection():
    try:
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"],
            insecure_mode=True,  # OCSP 검증 비활성화
            role='ANALYST'
        )
        return conn
    except Exception as e:
        st.error(f"Snowflake 연결 초기화 오류: {str(e)}")
        return None

# 연결 객체를 세션 상태에 저장
if 'snowflake_conn' not in st.session_state or st.session_state.snowflake_conn is None:
    st.session_state.snowflake_conn = init_snowflake_connection()

# Snowflake 쿼리 실행 함수
def get_snowflake_data(query, params=None):
    try:
        if st.session_state.snowflake_conn is None:
            st.error("Snowflake 연결이 초기화되지 않았습니다. 재시도 중...")
            st.session_state.snowflake_conn = init_snowflake_connection()
            if st.session_state.snowflake_conn is None:
                return None
        cur = st.session_state.snowflake_conn.cursor()
        cur.execute(query, params if params else ())
        df = cur.fetch_pandas_all()
        return df
    except Exception as e:
        st.error(f"Snowflake 쿼리 오류: {str(e)}")
        # 연결 오류 시 재초기화
        st.session_state.snowflake_conn = init_snowflake_connection()
        return None

st.title("백화점 소비/방문자 주가 분석 대시보드")

# 사이드바에 세련된 네비게이션 메뉴 추가
with st.sidebar:
    navigation = option_menu(
        menu_title="메뉴",
        options=[
            "홈",
            "소비 현황",
            "방문 현황",
            "주가 현황",
            "백테스팅 - 방문 현황",
            "수익률 비교 - 방문 현황",
            "백테스팅 - 소비 현황",
            "수익률 비교 - 소비 현황",
            "투자 인사이트 (LLM)"
        ],
        icons=[
            "house",
            "wallet",
            "person-walking",
            "building",
            "gear",
            "bar-chart-fill",
            "gear",
            "bar-chart-fill",
            "robot"
        ],
        menu_icon="menu-button-wide",
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#1c2526",
            },
            "icon": {
                "color": "#e0e0e0",
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "#e0e0e0",
                "--hover-color": "#2e3b3e",
            },
            "nav-link-selected": {
                "background-color": "#007bff",
                "color": "#ffffff",
            },
        },
    )

store_options = ['All', '신세계_강남', '더현대서울', '롯데백화점_본점']
selected_store = st.sidebar.selectbox("백화점 선택", store_options)
date_range = st.sidebar.slider(
    "시간 범위 선택",
    min_value=datetime(2021, 1, 1),
    max_value=datetime(2023, 12, 1),
    value=(datetime(2021, 1, 1), datetime(2023, 12, 1)),
    format="YYYY-MM"
)

# 메인 화면 (홈)
if navigation == "홈":
    st.markdown("""
        이 대시보드는 주요 백화점(더현대서울, 신세계_강남, 롯데백화점_본점)의 소비 패턴, 방문자 수, 주가 데이터를 분석하여 투자 인사이트를 제공합니다.\n
        Snowflake 데이터를 기반으로 소비 및 방문 추세를 시각화하고, LLM을 활용해 백테스팅 결과를 해석합니다.\n
        각 탭에서 소비/방문/주가 분석, 백테스팅, AI 기반 인사이트를 탐색해 보세요!
    """)

    # 백화점별 종목 리스트 (하드코딩)
    ticker_mapping = [
        {"TICKER": "004170", "DEPARTMENT_STORE": "신세계_강남", "COMPANY_NAME": "신세계"},
        {"TICKER": "031430", "DEPARTMENT_STORE": "신세계_강남", "COMPANY_NAME": "신세계인터내셔날"},
        {"TICKER": "031440", "DEPARTMENT_STORE": "신세계_강남", "COMPANY_NAME": "신세계푸드"},
        {"TICKER": "037710", "DEPARTMENT_STORE": "신세계_강남", "COMPANY_NAME": "광주신세계"},
        {"TICKER": "005440", "DEPARTMENT_STORE": "더현대서울", "COMPANY_NAME": "현대그린푸드"},
        {"TICKER": "020000", "DEPARTMENT_STORE": "더현대서울", "COMPANY_NAME": "한섬"},
        {"TICKER": "057050", "DEPARTMENT_STORE": "더현대서울", "COMPANY_NAME": "현대홈쇼핑"},
        {"TICKER": "069960", "DEPARTMENT_STORE": "더현대서울", "COMPANY_NAME": "현대백화점"},
        {"TICKER": "004990", "DEPARTMENT_STORE": "롯데백화점_본점", "COMPANY_NAME": "롯데지주"},
        {"TICKER": "011170", "DEPARTMENT_STORE": "롯데백화점_본점", "COMPANY_NAME": "롯데케미칼"},
        {"TICKER": "023530", "DEPARTMENT_STORE": "롯데백화점_본점", "COMPANY_NAME": "롯데쇼핑"},
        {"TICKER": "071840", "DEPARTMENT_STORE": "롯데백화점_본점", "COMPANY_NAME": "롯데하이마트"}
    ]

    # 백화점별 종목 그룹화
    shinsegae_stocks = [item for item in ticker_mapping if item["DEPARTMENT_STORE"] == "신세계_강남"]
    hyundai_stocks = [item for item in ticker_mapping if item["DEPARTMENT_STORE"] == "더현대서울"]
    lotte_stocks = [item for item in ticker_mapping if item["DEPARTMENT_STORE"] == "롯데백화점_본점"]

    # 종목 리스트 표시
    st.markdown("### 주요 백화점 관련 종목")
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container():
            st.markdown(
                f"<div style='background-color: #6db8de; padding: 10px; text-align: center;'>{shinsegae_stocks[0]['DEPARTMENT_STORE']}</div>",
                unsafe_allow_html=True
            )
            for stock in shinsegae_stocks:
                st.write(f"{stock['COMPANY_NAME']} ({stock['TICKER']})")

    with col2:
        with st.container():
            st.markdown(
                f"<div style='background-color: #6db8de; padding: 10px; text-align: center;'>{hyundai_stocks[0]['DEPARTMENT_STORE']}</div>",
                unsafe_allow_html=True
            )
            for stock in hyundai_stocks:
                st.write(f"{stock['COMPANY_NAME']} ({stock['TICKER']})")

    with col3:
        with st.container():
            st.markdown(
                f"<div style='background-color: #6db8de; padding: 10px; text-align: center;'>{lotte_stocks[0]['DEPARTMENT_STORE']}</div>",
                unsafe_allow_html=True
            )
            for stock in lotte_stocks:
                st.write(f"{stock['COMPANY_NAME']} ({stock['TICKER']})")
# Tab 1: Consumption Status (PROCESSED schema)
if navigation == "소비 현황":
    st.subheader("월별 소비 현황")
    query_consumption = f"""
    SELECT
        YEAR_MONTH,
        DEPARTMENT_STORE,
        AVG(DEPARTMENT_STORE_SPEND) AS DEPT_SPEND,
        AVG(DEPT_SPEND_GROWTH_RATE) AS DEPT_SPEND_GROWTH_RATE
    FROM FLOWCAST_DB.PROCESSED.CONSUMPTION_TRENDS
    WHERE YEAR_MONTH BETWEEN '{date_range[0].strftime('%Y-%m-%d')}' AND '{date_range[1].strftime('%Y-%m-%d')}'
    """
    if selected_store != 'All':
        query_consumption += f" AND DEPARTMENT_STORE = '{selected_store}'"
    query_consumption += " GROUP BY YEAR_MONTH, DEPARTMENT_STORE"

    with st.spinner("소비 데이터 로딩 중..."):
        df_consumption = get_snowflake_data(query_consumption)

    if df_consumption is not None and not df_consumption.empty:
        df_consumption['YEAR_MONTH'] = pd.to_datetime(df_consumption['YEAR_MONTH'])
        df_consumption = df_consumption.sort_values('YEAR_MONTH')
        df_consumption = df_consumption.dropna(subset=['DEPT_SPEND'])
        df_consumption['DEPT_SPEND_BILLION'] = df_consumption['DEPT_SPEND'].astype(float) / 100000000

        if selected_store != 'All':
            fig1 = px.line(
                df_consumption,
                x='YEAR_MONTH',
                y='DEPT_SPEND_BILLION',
                title=f'{selected_store} 소비 금액 추세'
            )
        else:
            fig1 = px.line(
                df_consumption,
                x='YEAR_MONTH',
                y='DEPT_SPEND_BILLION',
                color='DEPARTMENT_STORE',
                title='백화점별 소비 금액 추세'
            )
        fig1.update_layout(
            yaxis_title='소비 금액 (억 원)',
            yaxis=dict(range=[float(df_consumption['DEPT_SPEND_BILLION'].min()) * 0.95, float(df_consumption['DEPT_SPEND_BILLION'].max()) * 1.05]),
            legend_title='백화점'
        )
        st.plotly_chart(fig1)

        if selected_store != 'All':
            fig2 = px.line(
                df_consumption,
                x='YEAR_MONTH',
                y='DEPT_SPEND_GROWTH_RATE',
                title=f'{selected_store} 소비 증가율 추세'
            )
        else:
            fig2 = px.line(
                df_consumption,
                x='YEAR_MONTH',
                y='DEPT_SPEND_GROWTH_RATE',
                color='DEPARTMENT_STORE',
                title='백화점별 소비 증가율 추세'
            )
        fig2.update_layout(
            yaxis_title='소비 증가율 (%)',
            legend_title='백화점'
        )
        st.plotly_chart(fig2)
    else:
        st.warning("소비 데이터를 로드할 수 없거나 데이터가 없습니다.")

# Tab 2: Visitor Status (RAW schema)
elif navigation == "방문 현황":
    st.subheader("월별 방문 현황")
    # 월별 방문자 데이터를 가져오는 쿼리
    # date_range[1]을 월 마지막 날짜로 보정
    start_date = date_range[0]
    end_date = date_range[1]
    last_day = calendar.monthrange(end_date.year, end_date.month)[1]
    end_date = end_date.replace(day=last_day)

    query_visits = f"""
        SELECT
            DATE_TRUNC('MONTH', DATE) AS YEAR_MONTH,
            DEPARTMENT_STORE_NAME AS DEPARTMENT_STORE,
            SUM(COUNT) AS VISITOR_COUNT
        FROM FLOWCAST_DB.RAW.VISITS
        WHERE DATE BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
        """

    if selected_store != 'All':
        query_visits += f" AND DEPARTMENT_STORE_NAME = '{selected_store}'"
    query_visits += " GROUP BY DATE_TRUNC('MONTH', DATE), DEPARTMENT_STORE_NAME"

    try:
        with st.spinner("방문자 데이터 로딩 중..."):
            df_visits = get_snowflake_data(query_visits)

        if df_visits is not None and not df_visits.empty:
            df_visits['YEAR_MONTH'] = pd.to_datetime(df_visits['YEAR_MONTH'])
            df_visits = df_visits.sort_values('YEAR_MONTH')
            df_visits = df_visits.dropna(subset=['VISITOR_COUNT'])
            df_visits['VISITOR_COUNT'] = df_visits['VISITOR_COUNT'].astype(float)

            if selected_store != 'All':
                fig3 = px.line(
                    df_visits,
                    x='YEAR_MONTH',
                    y='VISITOR_COUNT',
                    title=f'{selected_store} 방문자 수 추세'
                )
            else:
                fig3 = px.line(
                    df_visits,
                    x='YEAR_MONTH',
                    y='VISITOR_COUNT',
                    color='DEPARTMENT_STORE',
                    title='백화점별 방문자 수 추세'
                )
            fig3.update_layout(
                yaxis_title='방문자 수',
                yaxis=dict(range=[float(df_visits['VISITOR_COUNT'].min()) * 0.95, float(df_visits['VISITOR_COUNT'].max()) * 1.05]),
                legend_title='백화점'
            )
            st.plotly_chart(fig3)
        else:
            st.warning("방문자 데이터를 로드할 수 없거나 데이터가 없습니다.")
    except snowflake.connector.errors.ProgrammingError as e:
        st.error(f"Snowflake 쿼리 실행 중 에러 발생: {str(e)}")

# Tab 3: Stock Price Status
elif navigation == "주가 현황":
    st.subheader("주가 현황")
    query_stock = f"""
    SELECT DISTINCT TICKER, COMPANY_NAME, DEPARTMENT_STORE
    FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA
    WHERE YEAR_MONTH BETWEEN '{date_range[0].strftime('%Y-%m-%d')}' AND '{date_range[1].strftime('%Y-%m-%d')}'
    """
    if selected_store != 'All':
        query_stock += f" AND DEPARTMENT_STORE = '{selected_store}'"

    try:
        df_tickers = get_snowflake_data(query_stock)
        if df_tickers is not None and not df_tickers.empty:
            ticker_display = df_tickers.apply(lambda x: f"{x['COMPANY_NAME']} ({x['TICKER']})", axis=1)
            ticker_map = dict(zip(ticker_display, df_tickers['TICKER']))
            selected_display = st.selectbox("종목 선택", ticker_display, key='stock_ticker')
            selected_ticker = ticker_map[selected_display]

            query_stock_monthly = f"""
            SELECT
                YEAR_MONTH,
                CLOSE_PRICE,
                PRICE_CHANGE
            FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA
            WHERE TICKER = '{selected_ticker}'
                AND YEAR_MONTH BETWEEN '{date_range[0].strftime('%Y-%m-%d')}' AND '{date_range[1].strftime('%Y-%m-%d')}'
            """
            query_stock_daily = f"""
            SELECT
                TRADE_DATE,
                OPEN_PRICE,
                HIGH_PRICE,
                LOW_PRICE,
                CLOSE_PRICE,
                VOLUME,
                FLUCTUATION
            FROM FLOWCAST_DB.RAW.STOCK_PRICES
            WHERE TICKER = '{selected_ticker}'
                AND TRADE_DATE BETWEEN '{date_range[0].strftime('%Y-%m-%d')}' AND '{date_range[1].strftime('%Y-%m-%d')}'
            """

            with st.spinner("주가 데이터 로딩 중..."):
                df_stock_monthly = get_snowflake_data(query_stock_monthly)
                df_stock_daily = get_snowflake_data(query_stock_daily)

            if df_stock_monthly is not None and not df_stock_monthly.empty:
                df_stock_monthly['YEAR_MONTH'] = pd.to_datetime(df_stock_monthly['YEAR_MONTH'])
                df_stock_monthly = df_stock_monthly.sort_values('YEAR_MONTH')
                df_stock_monthly = df_stock_monthly.dropna(subset=['CLOSE_PRICE'])

                fig4 = go.Figure()
                fig4.add_trace(
                    go.Scatter(
                        x=df_stock_monthly['YEAR_MONTH'],
                        y=df_stock_monthly['CLOSE_PRICE'],
                        name='종가',
                        line=dict(color='blue', width=2)
                    )
                )
                fig4.update_layout(
                    title=f'{selected_display} 주가 추세 (월봉)',
                    xaxis_title='월',
                    yaxis_title='주가 (원)',
                    template='plotly_dark',
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(autorange=True, fixedrange=False),
                    hovermode='x unified'
                )
                st.plotly_chart(fig4, use_container_width=True)

                fig5 = go.Figure()
                fig5.add_trace(
                    go.Bar(
                        x=df_stock_monthly['YEAR_MONTH'],
                        y=df_stock_monthly['PRICE_CHANGE'],
                        name='등락률',
                        marker_color=df_stock_monthly['PRICE_CHANGE'].apply(
                            lambda x: '#FF0000' if x < 0 else '#00FF00'),
                        marker_line_color='black',
                        marker_line_width=0.5,
                        opacity=1.0
                    )
                )
                fig5.update_layout(
                    title=f'{selected_display} 월별 주가 등락률',
                    xaxis_title='월',
                    yaxis_title='등락률 (%)',
                    template='plotly_dark',
                    showlegend=True,
                    bargap=0.2
                )
                st.plotly_chart(fig5, use_container_width=True)

            if df_stock_daily is not None and not df_stock_daily.empty:
                df_stock_daily['TRADE_DATE'] = pd.to_datetime(df_stock_daily['TRADE_DATE'])
                df_stock_daily = df_stock_daily.sort_values('TRADE_DATE')
                df_stock_daily = df_stock_daily.dropna(subset=['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE'])

                if 'VOLUME' in df_stock_daily.columns:
                    df_stock_daily['VOLUME_THOUSANDS'] = df_stock_daily['VOLUME'] / 1000
                else:
                    df_stock_daily['VOLUME_THOUSANDS'] = 0

                fig6 = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=(f'{selected_display} 주가 추세 (일봉)', '거래량'),
                    row_heights=[0.7, 0.3]
                )
                fig6.add_trace(
                    go.Candlestick(
                        x=df_stock_daily['TRADE_DATE'],
                        open=df_stock_daily['OPEN_PRICE'],
                        high=df_stock_daily['HIGH_PRICE'],
                        low=df_stock_daily['LOW_PRICE'],
                        close=df_stock_daily['CLOSE_PRICE'],
                        name='OHLC',
                        increasing_line_color='#00FF00',
                        decreasing_line_color='#FF0000'
                    ),
                    row=1, col=1
                )
                if 'VOLUME' in df_stock_daily.columns and df_stock_daily['VOLUME_THOUSANDS'].notna().any():
                    fig6.add_trace(
                        go.Bar(
                            x=df_stock_daily['TRADE_DATE'],
                            y=df_stock_daily['VOLUME_THOUSANDS'],
                            name='거래량',
                            marker_color=df_stock_daily['FLUCTUATION'].apply(
                                lambda x: '#00FF00' if x >= 0 else '#FF0000'),
                            marker_line_color='white',
                            marker_line_width=1.5,
                            opacity=1.0,
                            width=0.9,
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                if 'VOLUME' in df_stock_daily.columns:
                    max_volume = df_stock_daily['VOLUME_THOUSANDS'].max()
                    fig6.update_yaxes(
                        title_text='거래량 (천 주)',
                        range=[0, max_volume * 1.2] if max_volume > 0 else [0, 1],
                        row=2, col=1
                    )
                fig6.update_layout(
                    title=f'{selected_display} 주가 및 거래량 (일봉)',
                    xaxis_title='날짜',
                    yaxis_title='주가 (원)',
                    template='plotly_dark',
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(autorange=True, fixedrange=False),
                    hovermode='x unified',
                    height=800,
                    bargap=0.1
                )
                fig6.update_xaxes(rangeslider_visible=False, row=2, col=1)
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.warning("일봉 데이터를 로드할 수 없습니다.")
        else:
            st.warning("종목 데이터를 로드할 수 없습니다.")
    except snowflake.connector.errors.ProgrammingError as e:
        st.error(f"Snowflake 쿼리 실행 중 에러 발생: {str(e)}")

# Tab 4: Backtesting
elif navigation == "백테스팅 - 방문 현황":
    st.subheader("방문자 기반 최고 주가 전략")
    growth_threshold = st.slider("방문자 증가율 기준 (%)", 0.0, 20.0, 3.0, step=0.5)
    window_start = st.slider("윈도우 시작 (개월 후)", 1, 6, 1)
    window_end = st.slider("윈도우 종료 (개월 후)", 1, 12, 6)

    query_backtest = f"""
    WITH Visit_Growth AS (
        SELECT
            YEAR_MONTH,
            DEPARTMENT_STORE,
            TICKER,
            VISITOR_COUNT,
            100 * (
                VISITOR_COUNT - LAG(VISITOR_COUNT) OVER (
                    PARTITION BY DEPARTMENT_STORE, TICKER
                    ORDER BY YEAR_MONTH
                )
            ) / NULLIF(
                LAG(VISITOR_COUNT) OVER (
                    PARTITION BY DEPARTMENT_STORE, TICKER
                    ORDER BY YEAR_MONTH
                ), 0
            ) AS VISITOR_GROWTH_RATE,
            CLOSE_PRICE
        FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA
        WHERE YEAR_MONTH BETWEEN '{date_range[0].strftime('%Y-%m-%d')}' AND '{date_range[1].strftime('%Y-%m-%d')}'
    """
    if selected_store != 'All':
        query_backtest += f" AND DEPARTMENT_STORE = '{selected_store}'"
    query_backtest += f"""
    ),
    Signals AS (
        SELECT
            v.YEAR_MONTH,
            v.DEPARTMENT_STORE,
            v.TICKER,
            v.VISITOR_GROWTH_RATE,
            v.CLOSE_PRICE AS BUY_PRICE,
            h.TRADE_DATE,
            h.HIGH_PRICE,
            ROW_NUMBER() OVER (
                PARTITION BY v.TICKER, v.YEAR_MONTH
                ORDER BY h.HIGH_PRICE DESC, h.TRADE_DATE
            ) AS RN
        FROM Visit_Growth v
        LEFT JOIN FLOWCAST_DB.RAW.STOCK_PRICES h
            ON v.TICKER = h.TICKER
            AND h.TRADE_DATE BETWEEN 
                DATEADD(MONTH, {window_start}, v.YEAR_MONTH) 
                AND LAST_DAY(DATEADD(MONTH, {window_end}, v.YEAR_MONTH))
        WHERE v.VISITOR_GROWTH_RATE > {growth_threshold}
            AND v.VISITOR_GROWTH_RATE IS NOT NULL
    ),
    High_Price_Window AS (
        SELECT
            YEAR_MONTH,
            DEPARTMENT_STORE,
            TICKER,
            VISITOR_GROWTH_RATE,
            BUY_PRICE,
            TRADE_DATE AS HIGH_PRICE_DATE,
            HIGH_PRICE AS SELL_PRICE
        FROM Signals
        WHERE RN = 1
    )
    SELECT
        s.YEAR_MONTH,
        s.DEPARTMENT_STORE,
        s.TICKER,
        b.COMPANY_NAME,
        s.VISITOR_GROWTH_RATE,
        s.BUY_PRICE,
        s.SELL_PRICE,
        CASE
            WHEN s.SELL_PRICE IS NOT NULL AND s.BUY_PRICE > 0
            THEN 100 * (s.SELL_PRICE - s.BUY_PRICE) / s.BUY_PRICE
            ELSE NULL
        END AS RETURN_PERCENT,
        s.HIGH_PRICE_DATE
    FROM High_Price_Window s
    LEFT JOIN FLOWCAST_DB.ANALYTICS.BACKTEST_DATA b
        ON s.TICKER = b.TICKER
        AND s.YEAR_MONTH = b.YEAR_MONTH
        AND s.DEPARTMENT_STORE = b.DEPARTMENT_STORE
    ORDER BY s.YEAR_MONTH, s.TICKER;
    """

    try:
        with st.spinner("백테스팅 데이터 로딩 중..."):
            df_backtest = get_snowflake_data(query_backtest)

        if df_backtest is not None and not df_backtest.empty:
            df_backtest['YEAR_MONTH'] = pd.to_datetime(df_backtest['YEAR_MONTH']).dt.strftime('%Y-%m-%d')
            df_backtest['HIGH_PRICE_DATE'] = pd.to_datetime(df_backtest['HIGH_PRICE_DATE']).dt.strftime('%Y-%m-%d')
            df_backtest['VISITOR_GROWTH_RATE'] = df_backtest['VISITOR_GROWTH_RATE'].round(2)
            df_backtest['RETURN_PERCENT'] = df_backtest['RETURN_PERCENT'].round(2)
            df_backtest['BUY_PRICE'] = df_backtest['BUY_PRICE'].astype(float).round(0)
            df_backtest['SELL_PRICE'] = df_backtest['SELL_PRICE'].astype(float).round(0)

            def color_return(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}'

            styled_df = df_backtest.style.applymap(color_return, subset=['RETURN_PERCENT'])
            st.write(f"최고 주가 전략 결과 ({selected_store}, {window_start}~{window_end}개월 창, 방문 증가율 > {growth_threshold}%):", styled_df)

            with st.expander("디버깅 정보"):
                query_visit_growth = f"""
                SELECT
                    YEAR_MONTH,
                    TICKER,
                    VISITOR_COUNT,
                    100 * (
                        VISITOR_COUNT - LAG(VISITOR_COUNT) OVER (
                            PARTITION BY DEPARTMENT_STORE, TICKER
                            ORDER BY YEAR_MONTH
                        )
                    ) / NULLIF(
                        LAG(VISITOR_COUNT) OVER (
                            PARTITION BY DEPARTMENT_STORE, TICKER
                            ORDER BY YEAR_MONTH
                        ), 0
                    ) AS VISITOR_GROWTH_RATE
                FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA
                WHERE DEPARTMENT_STORE = '{selected_store}'
                    AND YEAR_MONTH BETWEEN '{date_range[0].strftime('%Y-%m-%d')}' AND '{date_range[1].strftime('%Y-%m-%d')}'
                """
                df_visit_growth = get_snowflake_data(query_visit_growth)
                st.write(f"방문자 증가율 데이터 ({selected_store}):", df_visit_growth)

                query_tickers = f"""
                SELECT DISTINCT TICKER, COMPANY_NAME
                FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA
                WHERE DEPARTMENT_STORE = '{selected_store}';
                """
                df_tickers = get_snowflake_data(query_tickers)
                st.write(f"{selected_store} 관련 주식 목록:", df_tickers)
        else:
            st.warning(f"백테스팅 데이터가 없습니다. 방문자 증가율 > {growth_threshold}% 조건, 데이터 범위, 또는 백화점 설정을 확인하세요.")
    except snowflake.connector.errors.ProgrammingError as e:
        st.error(f"Snowflake 쿼리 실행 중 에러 발생: {str(e)}")

# Tab 5: Short-Term vs Long-Term Return Comparison
elif navigation == "수익률 비교 - 방문 현황":
    st.subheader("단기 vs 장기 투자 수익률 비교")
    growth_threshold = st.slider("방문자 증가율 기준 (%)", 0.0, 20.0, 3.0, step=0.5, key='tab5_growth')
    short_window_start = 1
    short_window_end = 3
    long_window_start = 6
    long_window_end = 12

    def run_backtest(window_start, window_end, growth_threshold):
        query = f"""
        WITH Visit_Growth AS (
            SELECT
                YEAR_MONTH,
                DEPARTMENT_STORE,
                TICKER,
                VISITOR_COUNT,
                100 * (
                    VISITOR_COUNT - LAG(VISITOR_COUNT) OVER (
                        PARTITION BY DEPARTMENT_STORE, TICKER
                        ORDER BY YEAR_MONTH
                    )
                ) / NULLIF(
                    LAG(VISITOR_COUNT) OVER (
                        PARTITION BY DEPARTMENT_STORE, TICKER
                        ORDER BY YEAR_MONTH
                    ), 0
                ) AS VISITOR_GROWTH_RATE,
                CLOSE_PRICE
            FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA
            WHERE YEAR_MONTH BETWEEN '{date_range[0].strftime('%Y-%m-%d')}' AND '{date_range[1].strftime('%Y-%m-%d')}'
                AND VISITOR_COUNT IS NOT NULL
        """
        if selected_store != 'All':
            query += f" AND DEPARTMENT_STORE = '{selected_store}'"
        query += f"""
        ),
        Signals AS (
            SELECT
                v.YEAR_MONTH,
                v.DEPARTMENT_STORE,
                v.TICKER,
                v.VISITOR_GROWTH_RATE,
                v.CLOSE_PRICE AS BUY_PRICE,
                h.TRADE_DATE,
                h.HIGH_PRICE,
                ROW_NUMBER() OVER (
                    PARTITION BY v.TICKER, v.YEAR_MONTH
                    ORDER BY h.HIGH_PRICE DESC, h.TRADE_DATE
                ) AS RN
            FROM Visit_Growth v
            LEFT JOIN FLOWCAST_DB.RAW.STOCK_PRICES h
                ON v.TICKER = h.TICKER
                AND v.DEPARTMENT_STORE = h.DEPARTMENT_STORE
                AND h.TRADE_DATE BETWEEN 
                    DATEADD(MONTH, {window_start}, v.YEAR_MONTH) 
                    AND LAST_DAY(DATEADD(MONTH, {window_end}, v.YEAR_MONTH))
            WHERE v.VISITOR_GROWTH_RATE > {growth_threshold}
                AND v.VISITOR_GROWTH_RATE IS NOT NULL
        ),
        High_Price_Window AS (
            SELECT
                YEAR_MONTH,
                DEPARTMENT_STORE,
                TICKER,
                VISITOR_GROWTH_RATE,
                BUY_PRICE,
                TRADE_DATE AS HIGH_PRICE_DATE,
                HIGH_PRICE AS SELL_PRICE
            FROM Signals
            WHERE RN = 1
        )
        SELECT
            s.YEAR_MONTH,
            s.DEPARTMENT_STORE,
            s.TICKER,
            b.COMPANY_NAME,
            s.VISITOR_GROWTH_RATE,
            s.BUY_PRICE,
            s.SELL_PRICE,
            CASE
                WHEN s.SELL_PRICE IS NOT NULL AND s.BUY_PRICE > 0
                THEN 100 * (s.SELL_PRICE - s.BUY_PRICE) / s.BUY_PRICE
                ELSE NULL
            END AS RETURN_PERCENT,
            s.HIGH_PRICE_DATE
        FROM High_Price_Window s
        LEFT JOIN FLOWCAST_DB.ANALYTICS.BACKTEST_DATA b
            ON s.TICKER = b.TICKER
            AND s.YEAR_MONTH = b.YEAR_MONTH
            AND s.DEPARTMENT_STORE = b.DEPARTMENT_STORE
        ORDER BY s.YEAR_MONTH, s.TICKER;
        """
        return get_snowflake_data(query)

    try:
        with st.spinner("단기/장기 수익률 비교 데이터 로딩 중..."):
            df_short = run_backtest(short_window_start, short_window_end, growth_threshold)
            if df_short is not None and not df_short.empty:
                df_short['Strategy'] = 'Short-Term (1-3 months)'
                df_short['YEAR_MONTH'] = pd.to_datetime(df_short['YEAR_MONTH']).dt.strftime('%Y-%m-%d')
                df_short['HIGH_PRICE_DATE'] = pd.to_datetime(df_short['HIGH_PRICE_DATE']).dt.strftime('%Y-%m-%d')
                df_short['VISITOR_GROWTH_RATE'] = df_short['VISITOR_GROWTH_RATE'].round(2)
                df_short['RETURN_PERCENT'] = df_short['RETURN_PERCENT'].round(2)
                df_short['BUY_PRICE'] = df_short['BUY_PRICE'].astype(float).round(0)
                df_short['SELL_PRICE'] = df_short['SELL_PRICE'].astype(float).round(0)

            df_long = run_backtest(long_window_start, long_window_end, growth_threshold)
            if df_long is not None and not df_long.empty:
                df_long['Strategy'] = 'Long-Term (6-12 months)'
                df_long['YEAR_MONTH'] = pd.to_datetime(df_long['YEAR_MONTH']).dt.strftime('%Y-%m-%d')
                df_long['HIGH_PRICE_DATE'] = pd.to_datetime(df_long['HIGH_PRICE_DATE']).dt.strftime('%Y-%m-%d')
                df_long['VISITOR_GROWTH_RATE'] = df_long['VISITOR_GROWTH_RATE'].round(2)
                df_long['RETURN_PERCENT'] = df_long['RETURN_PERCENT'].round(2)
                df_long['BUY_PRICE'] = df_long['BUY_PRICE'].astype(float).round(0)
                df_long['SELL_PRICE'] = df_long['SELL_PRICE'].astype(float).round(0)

        if df_short is not None and not df_short.empty and df_long is not None and not df_long.empty:
            df_combined = pd.concat([df_short, df_long]).reset_index(drop=True)

            fig_box = px.box(
                df_combined,
                x='COMPANY_NAME',
                y='RETURN_PERCENT',
                color='Strategy',
                title=f'단기 (1-3개월) vs 장기 (6-12개월) 투자 수익률 비교 ({selected_store})',
                labels={'RETURN_PERCENT': '수익률 (%)', 'COMPANY_NAME': '회사명'}
            )
            st.plotly_chart(fig_box)
        else:
            st.warning(f"단기 또는 장기 백테스팅 데이터가 없습니다. 방문자 증가율 > {growth_threshold}% 조건, 데이터 범위, 또는 백화점 설정을 확인하세요.")
    except snowflake.connector.errors.ProgrammingError as e:
        st.error(f"Snowflake 쿼리 실행 중 에러 발생: {str(e)}")

# Tab 6: Backtesting - Consumption Status
elif navigation == "백테스팅 - 소비 현황":
    st.subheader("소비 기반 최고 주가 전략")
    growth_threshold = st.slider("소비 증가율 기준 (%)", 0.0, 20.0, 3.0, step=0.5, key='tab6_growth')
    window_start = st.slider("윈도우 시작 (개월 후)", 1, 6, 1, key='tab6_start')
    window_end = st.slider("윈도우 종료 (개월 후)", 1, 12, 6, key='tab6_end')

    query_backtest = f"""
    WITH Spend_Growth AS (
        SELECT
            YEAR_MONTH,
            DEPARTMENT_STORE,
            TICKER,
            DEPT_SPEND_GROWTH_RATE,
            CLOSE_PRICE
        FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA
        WHERE YEAR_MONTH BETWEEN '{date_range[0].strftime('%Y-%m-%d')}' AND '{date_range[1].strftime('%Y-%m-%d')}'
            AND DEPT_SPEND_GROWTH_RATE IS NOT NULL
    """
    if selected_store != 'All':
        query_backtest += f" AND DEPARTMENT_STORE = '{selected_store}'"
    query_backtest += f"""
    ),
    Signals AS (
        SELECT
            v.YEAR_MONTH,
            v.DEPARTMENT_STORE,
            v.TICKER,
            v.DEPT_SPEND_GROWTH_RATE,
            v.CLOSE_PRICE AS BUY_PRICE,
            h.TRADE_DATE,
            h.HIGH_PRICE,
            ROW_NUMBER() OVER (
                PARTITION BY v.TICKER, v.YEAR_MONTH
                ORDER BY h.HIGH_PRICE DESC, h.TRADE_DATE
            ) AS RN
        FROM Spend_Growth v
        LEFT JOIN FLOWCAST_DB.RAW.STOCK_PRICES h
            ON v.TICKER = h.TICKER
            AND v.DEPARTMENT_STORE = h.DEPARTMENT_STORE
            AND h.TRADE_DATE BETWEEN 
                DATEADD(MONTH, {window_start}, v.YEAR_MONTH) 
                AND LAST_DAY(DATEADD(MONTH, {window_end}, v.YEAR_MONTH))
        WHERE v.DEPT_SPEND_GROWTH_RATE > {growth_threshold}
    ),
    High_Price_Window AS (
        SELECT
            YEAR_MONTH,
            DEPARTMENT_STORE,
            TICKER,
            DEPT_SPEND_GROWTH_RATE,
            BUY_PRICE,
            TRADE_DATE AS HIGH_PRICE_DATE,
            HIGH_PRICE AS SELL_PRICE
        FROM Signals
        WHERE RN = 1
    )
    SELECT
        s.YEAR_MONTH,
        s.DEPARTMENT_STORE,
        s.TICKER,
        b.COMPANY_NAME,
        s.DEPT_SPEND_GROWTH_RATE,
        s.BUY_PRICE,
        s.SELL_PRICE,
        CASE
            WHEN s.SELL_PRICE IS NOT NULL AND s.BUY_PRICE > 0
            THEN 100 * (s.SELL_PRICE - s.BUY_PRICE) / s.BUY_PRICE
            ELSE NULL
        END AS RETURN_PERCENT,
        s.HIGH_PRICE_DATE
    FROM High_Price_Window s
    LEFT JOIN FLOWCAST_DB.ANALYTICS.BACKTEST_DATA b
        ON s.TICKER = b.TICKER
        AND s.YEAR_MONTH = b.YEAR_MONTH
        AND s.DEPARTMENT_STORE = b.DEPARTMENT_STORE
    ORDER BY s.YEAR_MONTH, s.TICKER;
    """

    try:
        with st.spinner("백테스팅 데이터 로딩 중..."):
            df_backtest = get_snowflake_data(query_backtest)

        if df_backtest is not None and not df_backtest.empty:
            df_backtest['YEAR_MONTH'] = pd.to_datetime(df_backtest['YEAR_MONTH']).dt.strftime('%Y-%m-%d')
            df_backtest['HIGH_PRICE_DATE'] = pd.to_datetime(df_backtest['HIGH_PRICE_DATE']).dt.strftime('%Y-%m-%d')
            df_backtest['DEPT_SPEND_GROWTH_RATE'] = df_backtest['DEPT_SPEND_GROWTH_RATE'].round(2)
            df_backtest['RETURN_PERCENT'] = df_backtest['RETURN_PERCENT'].round(2)
            df_backtest['BUY_PRICE'] = df_backtest['BUY_PRICE'].astype(float).round(0)
            df_backtest['SELL_PRICE'] = df_backtest['SELL_PRICE'].astype(float).round(0)

            def color_return(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}'

            styled_df = df_backtest.style.applymap(color_return, subset=['RETURN_PERCENT'])
            st.write(f"소비 기반 최고 주가 전략 결과 ({selected_store}, {window_start}~{window_end}개월 창, 소비 증가율 > {growth_threshold}%):", styled_df)
        else:
            st.warning(f"백테스팅 데이터가 없습니다. 소비 증가율 > {growth_threshold}% 조건, 데이터 범위, 또는 백화점 설정을 확인하세요.")
    except snowflake.connector.errors.ProgrammingError as e:
        st.error(f"Snowflake 쿼리 실행 중 에러 발생: {str(e)}")

# Tab 7: Short-Term vs Long-Term Return Comparison - Consumption Status
elif navigation == "수익률 비교 - 소비 현황":
    st.subheader("소비 기반 단기 vs 장기 투자 수익률 비교")
    growth_threshold = st.slider("소비 증가율 기준 (%)", 0.0, 20.0, 3.0, step=0.5, key='tab7_growth')
    short_window_start = 1
    short_window_end = 3
    long_window_start = 6
    long_window_end = 12

    def run_backtest(window_start, window_end, growth_threshold):
        query = f"""
        WITH Spend_Growth AS (
            SELECT
                YEAR_MONTH,
                DEPARTMENT_STORE,
                TICKER,
                DEPT_SPEND_GROWTH_RATE,
                CLOSE_PRICE
            FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA
            WHERE YEAR_MONTH BETWEEN '{date_range[0].strftime('%Y-%m-%d')}' AND '{date_range[1].strftime('%Y-%m-%d')}'
                AND DEPT_SPEND_GROWTH_RATE IS NOT NULL
        """
        if selected_store != 'All':
            query += f" AND DEPARTMENT_STORE = '{selected_store}'"
        query += f"""
        ),
        Signals AS (
            SELECT
                v.YEAR_MONTH,
                v.DEPARTMENT_STORE,
                v.TICKER,
                v.DEPT_SPEND_GROWTH_RATE,
                v.CLOSE_PRICE AS BUY_PRICE,
                h.TRADE_DATE,
                h.HIGH_PRICE,
                ROW_NUMBER() OVER (
                    PARTITION BY v.TICKER, v.YEAR_MONTH
                    ORDER BY h.HIGH_PRICE DESC, h.TRADE_DATE
                ) AS RN
            FROM Spend_Growth v
            LEFT JOIN FLOWCAST_DB.RAW.STOCK_PRICES h
                ON v.TICKER = h.TICKER
                AND v.DEPARTMENT_STORE = h.DEPARTMENT_STORE
                AND h.TRADE_DATE BETWEEN 
                    DATEADD(MONTH, {window_start}, v.YEAR_MONTH) 
                    AND LAST_DAY(DATEADD(MONTH, {window_end}, v.YEAR_MONTH))
            WHERE v.DEPT_SPEND_GROWTH_RATE > {growth_threshold}
        ),
        High_Price_Window AS (
            SELECT
                YEAR_MONTH,
                DEPARTMENT_STORE,
                TICKER,
                DEPT_SPEND_GROWTH_RATE,
                BUY_PRICE,
                TRADE_DATE AS HIGH_PRICE_DATE,
                HIGH_PRICE AS SELL_PRICE
            FROM Signals
            WHERE RN = 1
        )
        SELECT
            s.YEAR_MONTH,
            s.DEPARTMENT_STORE,
            s.TICKER,
            b.COMPANY_NAME,
            s.DEPT_SPEND_GROWTH_RATE,
            s.BUY_PRICE,
            s.SELL_PRICE,
            CASE
                WHEN s.SELL_PRICE IS NOT NULL AND s.BUY_PRICE > 0
                THEN 100 * (s.SELL_PRICE - s.BUY_PRICE) / s.BUY_PRICE
                ELSE NULL
            END AS RETURN_PERCENT,
            s.HIGH_PRICE_DATE
        FROM High_Price_Window s
        LEFT JOIN FLOWCAST_DB.ANALYTICS.BACKTEST_DATA b
            ON s.TICKER = b.TICKER
            AND s.YEAR_MONTH = b.YEAR_MONTH
            AND s.DEPARTMENT_STORE = b.DEPARTMENT_STORE
        ORDER BY s.YEAR_MONTH, s.TICKER;
        """
        return get_snowflake_data(query)

    try:
        with st.spinner("단기/장기 수익률 비교 데이터 로딩 중..."):
            df_short = run_backtest(short_window_start, short_window_end, growth_threshold)
            if df_short is not None and not df_short.empty:
                df_short['Strategy'] = 'Short-Term (1-3 months)'
                df_short['YEAR_MONTH'] = pd.to_datetime(df_short['YEAR_MONTH']).dt.strftime('%Y-%m-%d')
                df_short['HIGH_PRICE_DATE'] = pd.to_datetime(df_short['HIGH_PRICE_DATE']).dt.strftime('%Y-%m-%d')
                df_short['DEPT_SPEND_GROWTH_RATE'] = df_short['DEPT_SPEND_GROWTH_RATE'].round(2)
                df_short['RETURN_PERCENT'] = df_short['RETURN_PERCENT'].round(2)
                df_short['BUY_PRICE'] = df_short['BUY_PRICE'].astype(float).round(0)
                df_short['SELL_PRICE'] = df_short['SELL_PRICE'].astype(float).round(0)

            df_long = run_backtest(long_window_start, long_window_end, growth_threshold)
            if df_long is not None and not df_long.empty:
                df_long['Strategy'] = 'Long-Term (6-12 months)'
                df_long['YEAR_MONTH'] = pd.to_datetime(df_long['YEAR_MONTH']).dt.strftime('%Y-%m-%d')
                df_long['HIGH_PRICE_DATE'] = pd.to_datetime(df_long['HIGH_PRICE_DATE']).dt.strftime('%Y-%m-%d')
                df_long['DEPT_SPEND_GROWTH_RATE'] = df_long['DEPT_SPEND_GROWTH_RATE'].round(2)
                df_long['RETURN_PERCENT'] = df_long['RETURN_PERCENT'].round(2)
                df_long['BUY_PRICE'] = df_long['BUY_PRICE'].astype(float).round(0)
                df_long['SELL_PRICE'] = df_long['SELL_PRICE'].astype(float).round(0)

        if df_short is not None and not df_short.empty and df_long is not None and not df_long.empty:
            df_combined = pd.concat([df_short, df_long]).reset_index(drop=True)

            fig_box = px.box(
                df_combined,
                x='COMPANY_NAME',
                y='RETURN_PERCENT',
                color='Strategy',
                title=f'소비 기반 단기 (1-3개월) vs 장기 (6-12개월) 투자 수익률 비교 ({selected_store})',
                labels={'RETURN_PERCENT': '수익률 (%)', 'COMPANY_NAME': '회사명'}
            )
            st.plotly_chart(fig_box)
        else:
            st.warning(f"단기 또는 장기 백테스팅 데이터가 없습니다. 소비 증가율 > {growth_threshold}% 조건, 데이터 범위, 또는 백화점 설정을 확인하세요.")
    except snowflake.connector.errors.ProgrammingError as e:
        st.error(f"Snowflake 쿼리 실행 중 에러 발생: {str(e)}")

#  LLM을 사용한 투자 인사이트
if navigation == "투자 인사이트 (LLM)":
    st.subheader("투자 인사이트 (LLM)")

    # 채팅 기록 및 캐시 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "data_cache" not in st.session_state:
        st.session_state.data_cache = {}

    # 백화점별 티커 매핑 정의
    ticker_mapping = [
        {"TICKER": "004170", "DEPARTMENT_STORE": "신세계_강남", "COMPANY_NAME": "신세계", "INDUSTRY": "소매"},
        {"TICKER": "004990", "DEPARTMENT_STORE": "롯데백화점_본점", "COMPANY_NAME": "롯데지주", "INDUSTRY": "지주"},
        {"TICKER": "005440", "DEPARTMENT_STORE": "더현대서울", "COMPANY_NAME": "현대그린푸드", "INDUSTRY": "식품"},
        {"TICKER": "011170", "DEPARTMENT_STORE": "롯데백화점_본점", "COMPANY_NAME": "롯데케미칼", "INDUSTRY": "화학"},
        {"TICKER": "020000", "DEPARTMENT_STORE": "더현대서울", "COMPANY_NAME": "한섬", "INDUSTRY": "패션"},
        {"TICKER": "023530", "DEPARTMENT_STORE": "롯데백화점_본점", "COMPANY_NAME": "롯데쇼핑", "INDUSTRY": "소매"},
        {"TICKER": "031430", "DEPARTMENT_STORE": "신세계_강남", "COMPANY_NAME": "신세계인터내셔날", "INDUSTRY": "패션"},
        {"TICKER": "031440", "DEPARTMENT_STORE": "신세계_강남", "COMPANY_NAME": "신세계푸드", "INDUSTRY": "식품"},
        {"TICKER": "037710", "DEPARTMENT_STORE": "신세계_강남", "COMPANY_NAME": "광주신세계", "INDUSTRY": "소매"},
        {"TICKER": "057050", "DEPARTMENT_STORE": "더현대서울", "COMPANY_NAME": "현대홈쇼핑", "INDUSTRY": "홈쇼핑"},
        {"TICKER": "069960", "DEPARTMENT_STORE": "더현대서울", "COMPANY_NAME": "현대백화점", "INDUSTRY": "소매"},
        {"TICKER": "071840", "DEPARTMENT_STORE": "롯데백화점_본점", "COMPANY_NAME": "롯데하이마트", "INDUSTRY": "전자제품"}
    ]

    # 백화점별 맥락 정의
    store_contexts = {
        "신세계_강남": "고급 소비 중심의 프리미엄 백화점",
        "롯데백화점_본점": "대중적이고 다양한 소비층을 대상으로 한 백화점",
        "더현대서울": "현대적이고 트렌디한 소비를 지향하는 백화점"
    }

    # 사용자 입력
    store = st.selectbox("백화점 선택", ["신세계_강남", "롯데백화점_본점", "더현대서울"], key="store")
    ticker_options = [f"{item['TICKER']} ({item['COMPANY_NAME']})" for item in ticker_mapping if item['DEPARTMENT_STORE'] == store]
    if not ticker_options:
        st.warning(f"{store}에 대한 티커가 없습니다. 다른 백화점을 선택하세요.")
        ticker = None
        ticker_value = None
    else:
        ticker = st.selectbox("티커 선택", ticker_options, key="ticker")
        ticker_value = ticker.split(" ")[0]  # 예: "031430"

    data_type = st.radio("데이터 유형", ["소비 증가율", "방문 증가율"], key="data_type")
    # date_range 처리
    start_date = date_range[0].strftime('%Y-%m-%d') if len(date_range) > 0 else '2021-01-01'
    end_date = date_range[1].strftime('%Y-%m-%d') if len(date_range) > 1 else '2023-12-31'

    # 회사명 및 산업 맥락
    company_name = next((item['COMPANY_NAME'] for item in ticker_mapping if item['TICKER'] == ticker_value), "Unknown") if ticker_value else "Unknown"
    industry_context = next((item['INDUSTRY'] for item in ticker_mapping if item['TICKER'] == ticker_value), "Unknown") if ticker_value else "Unknown"
    store_context = store_contexts.get(store, "일반 백화점")

    # 질문 입력
    st.markdown("**질문을 입력하세요!** (예: 이 주식에 투자해야 할까?)")
    question_options = [
        "소비/방문 트렌드와 주가 관계를 요약해줘",
        "이 티커에 투자해야 할까?",
        "소비 증가율이 높을 때 주가와 거래량은 어떻게 변해?"
    ]
    selected_question = st.selectbox("추천 질문 선택 (또는 직접 입력)", [""] + question_options, key="question_select")
    user_question = st.chat_input("질문을 입력하세요...", key="question") or selected_question

    # 채팅 기록 표시
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])

    # 질문 처리
    if user_question and ticker_value:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "message": user_question})

        with st.spinner("분석 중..."):
            # 캐시 키 생성
            cache_key = f"{store}_{ticker_value}_{data_type}_{start_date}_{end_date}"

            # 캐시 확인 및 빈 캐시 제거
            if cache_key in st.session_state.data_cache and (
                    st.session_state.data_cache[cache_key] is None or st.session_state.data_cache[cache_key].empty):
                del st.session_state.data_cache[cache_key]

            # 캐시 확인
            if cache_key not in st.session_state.data_cache:
                growth_col = 'DEPT_SPEND_GROWTH_RATE' if data_type == '소비 증가율' else 'VISITOR_GROWTH_RATE'
                query = f"""
                    WITH Growth AS (
                        SELECT
                            b.YEAR_MONTH,
                            b.DEPARTMENT_STORE,
                            b.TICKER,
                            b.COMPANY_NAME,
                            b.DEPT_SPEND_GROWTH_RATE,
                            b.VISITOR_COUNT,
                            b.CLOSE_PRICE AS BUY_PRICE,
                            100 * (
                                b.VISITOR_COUNT - LAG(b.VISITOR_COUNT) OVER (
                                    PARTITION BY b.DEPARTMENT_STORE, b.TICKER
                                    ORDER BY b.YEAR_MONTH
                                )
                            ) / NULLIF(
                                LAG(b.VISITOR_COUNT) OVER (
                                    PARTITION BY b.DEPARTMENT_STORE, b.TICKER
                                    ORDER BY b.YEAR_MONTH
                                ), 0
                            ) AS VISITOR_GROWTH_RATE
                        FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA b
                        WHERE b.DEPARTMENT_STORE = %(store)s
                            AND b.TICKER = %(ticker)s
                            AND b.YEAR_MONTH BETWEEN %(start_date)s AND %(end_date)s
                    ),
                    Backtest AS (
                        SELECT
                            g.YEAR_MONTH,
                            g.DEPARTMENT_STORE,
                            g.TICKER,
                            g.COMPANY_NAME,
                            g.DEPT_SPEND_GROWTH_RATE AS DEPT_SPEND_GROWTH_RATE,
                            g.VISITOR_GROWTH_RATE AS VISITOR_GROWTH_RATE,
                            g.BUY_PRICE,
                            MAX(s.HIGH_PRICE) AS SELL_PRICE,
                            AVG(s.VOLUME) AS AVG_VOLUME
                        FROM Growth g
                        LEFT JOIN FLOWCAST_DB.RAW.STOCK_PRICES s
                            ON g.TICKER = s.TICKER
                            AND DATE_TRUNC('MONTH', s.TRADE_DATE) = g.YEAR_MONTH
                        WHERE g.{growth_col} > 0
                        GROUP BY
                            g.YEAR_MONTH,
                            g.DEPARTMENT_STORE,
                            g.TICKER,
                            g.COMPANY_NAME,
                            g.DEPT_SPEND_GROWTH_RATE,
                            g.VISITOR_GROWTH_RATE,
                            g.BUY_PRICE
                    )
                    SELECT
                        YEAR_MONTH,
                        DEPARTMENT_STORE,
                        TICKER,
                        COMPANY_NAME,
                        DEPT_SPEND_GROWTH_RATE,
                        VISITOR_GROWTH_RATE,
                        BUY_PRICE,
                        SELL_PRICE,
                        AVG_VOLUME,
                        CASE
                            WHEN SELL_PRICE IS NOT NULL AND BUY_PRICE > 0
                            THEN 100 * (SELL_PRICE - BUY_PRICE) / BUY_PRICE
                            ELSE NULL
                        END AS RETURN_PERCENT
                    FROM Backtest
                    ORDER BY YEAR_MONTH DESC
                """
                params = {
                    "store": store,
                    "ticker": ticker_value,
                    "start_date": start_date,
                    "end_date": end_date
                }
                df = get_snowflake_data(query, params)
                st.session_state.data_cache[cache_key] = df
            else:
                df = st.session_state.data_cache[cache_key]

            if df is not None and not df.empty:
                # 데이터 요약
                avg_growth = df["DEPT_SPEND_GROWTH_RATE"].mean() if data_type == "소비 증가율" else df[
                    "VISITOR_GROWTH_RATE"].mean()
                avg_return = df["RETURN_PERCENT"].mean()
                avg_volume = df["AVG_VOLUME"].mean()
                max_return = df["RETURN_PERCENT"].max()
                min_return = df["RETURN_PERCENT"].min()

                # 데이터프레임 표시
                st.write(f"{store} - {ticker} 백테스팅 결과 ({data_type} 증가 시점, {start_date} ~ {end_date}):")
                df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"]).dt.strftime("%Y-%m-%d")
                df["DEPT_SPEND_GROWTH_RATE"] = df["DEPT_SPEND_GROWTH_RATE"].round(2)
                df["VISITOR_GROWTH_RATE"] = df["VISITOR_GROWTH_RATE"].round(2)
                df["RETURN_PERCENT"] = df["RETURN_PERCENT"].round(2)
                df["AVG_VOLUME"] = df["AVG_VOLUME"].round(0)
                st.dataframe(
                    df[["YEAR_MONTH", "DEPT_SPEND_GROWTH_RATE", "VISITOR_GROWTH_RATE", "BUY_PRICE", "SELL_PRICE",
                        "RETURN_PERCENT", "AVG_VOLUME"]]
                )

                # 2021-10-01 데이터 디버깅
                if not df.empty:
                    df_2021_10 = df[df['YEAR_MONTH'] == '2021-10-01']
                    if not df_2021_10.empty:
                        st.write(
                            f"디버깅: 2021-10-01 데이터 - 소비 증가율: {df_2021_10['DEPT_SPEND_GROWTH_RATE'].iloc[0]}%, 수익률: {df_2021_10['RETURN_PERCENT'].iloc[0]}%, 거래량: {df_2021_10['AVG_VOLUME'].iloc[0]}")
                    else:
                        st.write("디버깅: 2021-10-01 데이터 없음")

                # 데이터프레임을 CSV로 변환
                data_csv = df[["YEAR_MONTH", "DEPT_SPEND_GROWTH_RATE", "VISITOR_GROWTH_RATE", "RETURN_PERCENT",
                               "AVG_VOLUME"]].to_csv(index=False)

                print (data_csv)

                # LLM 프롬프트
                prompt = f"""
                {store}의 {ticker} ({company_name})에 대한 백테스팅 데이터를 분석하세요.  
                - **데이터**: 아래는 {data_type}가 증가한 시점({data_type} > 0)의 월별 데이터 (CSV 형식):
                {data_csv}
                - **요약**: 평균 {data_type} {avg_growth:.2f}%, 평균 수익률 {avg_return:.2f}%, 최대 수익률 {max_return:.2f}%, 최소 수익률 {min_return:.2f}%, 평균 거래량 {avg_volume:.0f}.  
                - **기간**: {start_date} ~ {end_date}.  
                - **맥락**: {store}은 {store_context}, {company_name}은 {industry_context} 산업에 속함.  
                - **역할**: 주식 투자 전문가로서, {data_type}가 증가한 시점에 초점을 맞춘 정량적이고 실용적인 인사이트 제공.  

                **응답 구조**:  
                1. **요약**: {data_type} 증가 시 주가 수익률의 상관관계 (예: "{data_type} 5% 이상 시 평균 수익률 X%").  
                2. **트렌드**: {data_type}가 5% 이상 증가한 달의 수익률과 거래량 변화 (예: "2021-10-01: Y% 상승, 거래량 Z% 증가"). 데이터에 해당 월이 없으면 "조건에 맞는 데이터 없음" 명시.  
                3. **투자 전략**: {data_type} 증가 기반 매수/매도 전략 (예: "{data_type} 3% 초과 시 매수").  
                4. **주의점**: 데이터 한계 (예: "증가 시점 데이터만 분석")와 추가 고려 사항 (예: "{industry_context} 산업의 변동성").  

                **사용자 질문**: "{user_question}"  
                - 질문에 명확히 답변, 모호하면 {data_type} 증가 시점 데이터로 해석 (예: "투자해야 할까?" → "{data_type} 5% 이상 시 투자 권장").  

                **지침**:  
                - 각 섹션 2-3문장, 데이터에서 추출한 숫자 포함.  
                - {industry_context}를 고려해 산업별 요인 반영.  
                - 데이터 부족 시 다른 티커/기간 제안.  
                - 트렌드 분석은 제공된 CSV 데이터에 기반해야 하며, 존재하지 않는 날짜/수치(예: 2021-10-01, 15.6%)는 생성 금지.  
                """
                explanation_query = """
                    SELECT SNOWFLAKE.CORTEX.COMPLETE('CLAUDE-3-5-SONNET', %(prompt)s) AS EXPLANATION
                """
                explanation_df = get_snowflake_data(explanation_query, params={"prompt": prompt})

                if explanation_df is not None and not explanation_df.empty:
                    explanation = explanation_df["EXPLANATION"][0]
                    with st.chat_message("assistant"):
                        st.markdown(explanation)
                    st.session_state.chat_history.append({"role": "assistant", "message": explanation})
                else:
                    with st.chat_message("assistant"):
                        st.error("분석에 실패했습니다. Snowflake 연결을 확인하세요.")
                    st.session_state.chat_history.append({"role": "assistant", "message": "분석 실패"})
            else:
                with st.chat_message("assistant"):
                    st.warning(f"{store}의 {ticker_value} 데이터가 없습니다. 다른 백화점/티커를 선택하거나 기간을 조정하세요!")
                    debug_query = """
                        SELECT COUNT(*) AS record_count
                        FROM FLOWCAST_DB.ANALYTICS.BACKTEST_DATA
                        WHERE DEPARTMENT_STORE = %(store)s AND TICKER = %(ticker)s
                            AND YEAR_MONTH BETWEEN %(start_date)s AND %(end_date)s
                    """
                    debug_df = get_snowflake_data(debug_query, params={"store": store, "ticker": ticker_value,
                                                                       "start_date": start_date, "end_date": end_date})
                    if debug_df is not None:
                        st.write(
                            f"디버깅: BACKTEST_DATA에 {store} 및 {ticker_value} 데이터 레코드 수: {debug_df['RECORD_COUNT'].iloc[0]}")
                    st.session_state.chat_history.append({"role": "assistant", "message": f"{store}의없음"})

