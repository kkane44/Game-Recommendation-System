# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Board Game Recommender",
    page_icon="🎲",
    layout="wide"
)

# Load and process data once
@st.cache_data
def load_and_process_data():
    """Load CSV and perform all preprocessing"""
    try:
        df = pd.read_csv('bgg_dataset.csv', sep=';')
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None, None, None

    # Clean data
    df.loc[13984, ['Year Published']] = 1855
    df['Year Published'] = df['Year Published'].astype(int)
    df['Rating Average'] = df['Rating Average'].replace({',': '.'}, regex=True).astype(float)
    df['Complexity Average'] = df['Complexity Average'].replace({',': '.'}, regex=True).astype(float)
    df['Mechanics'] = df['Mechanics'].fillna('Not Specified')
    df['Domains'] = df['Domains'].fillna('Not Specified')

    # Clean mechanics text
    df['Mechanics'] = df['Mechanics'].apply(lambda text: re.sub('[^a-zA-Z0-9 ,]', '', str(text)).replace("/", ","))

    # Create dummy variables
    all_dummy_variables = pd.get_dummies(
        df['Mechanics'].str.split(', ', expand=True).stack(),
        prefix='mechanic'
    ).groupby(level=0).sum()

    # Create feature matrix
    df1 = pd.concat([df, all_dummy_variables], axis=1)

    # Scale continuous features
    X_continuous = df1[['Year Published', 'Min Players', 'Max Players', 'Play Time', 'Min Age', 'Rating Average', 'Complexity Average']]
    scaler = StandardScaler()
    X_cont_scaled = scaler.fit_transform(X_continuous)
    X_cont_scaled_df = pd.DataFrame(X_cont_scaled, columns=X_continuous.columns)

    # Combine scaled with dummy variables
    X_combined = pd.concat([X_cont_scaled_df, all_dummy_variables], axis=1)

    # Perform PCA
    optimal_pca = PCA(n_components=8)
    X_pca = optimal_pca.fit_transform(X_combined)

    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(X_pca)

    return df, cosine_sim_matrix, df['Name'].values, df['Name'].str.lower().values

# Load data
df, cosine_similarity_matrix, all_titles, all_titles_lower = load_and_process_data()

if df is None:
    st.error("Failed to load data. Please check your CSV file.")
    st.stop()

# Helper functions
@st.cache_data
def find_similar_titles(input_title, _all_titles=all_titles):
    """Find similar game titles to exclude (expansions, variants)"""
    similar_titles = []
    input_lower = input_title.lower()
    for title in _all_titles:
        title_lower = title.lower()
        if (input_lower in title_lower and input_lower != title_lower) or "expansion" in title_lower:
            similar_titles.append(title)
    return similar_titles

@st.cache_data
def search_games(query, _all_titles=all_titles):
    """Search for games matching the query"""
    if not query or len(query) < 2:
        return []
    query_lower = query.lower()
    matches = [title for title in _all_titles if query_lower in title.lower()]
    return sorted(matches)[:10]  # Return top 10 matches

@st.cache_data
def get_recommendations(names, _cosine_sim_matrix=cosine_similarity_matrix, _df=df,
                       player_count=None, complexity_range=None, max_play_time=None, min_rating=None):
    """Get game recommendations based on similarity with optional filters

    Args:
        names: Single game name (str) or list of game names
    """
    # Handle single game or multiple games
    if isinstance(names, str):
        names = [names]

    names = [name.strip() for name in names if name.strip()]

    if not names:
        return None, None, "Please enter at least one game name."

    # Find all game indices
    game_indices = []
    game_names_found = []
    not_found = []

    for name in names:
        name_lower = name.lower().strip()
        matching_games = _df[_df['Name'].str.lower() == name_lower]

        if matching_games.empty:
            not_found.append(name)
        else:
            game_indices.append(matching_games.index[0])
            game_names_found.append(matching_games.iloc[0]['Name'])

    if not game_indices:
        partial_matches = _df[_df['Name'].str.lower().str.contains(names[0].lower(), na=False)]
        if not partial_matches.empty:
            return None, None, f"Game '{names[0]}' not found. Did you mean: {', '.join(partial_matches['Name'].head(5).tolist())}?"
        return None, None, f"No games found. Please check spelling."

    # Calculate average similarity scores across all input games
    if len(game_indices) == 1:
        combined_scores = _cosine_sim_matrix[game_indices[0]]
    else:
        # Average the similarity scores from all input games
        combined_scores = np.mean([_cosine_sim_matrix[idx] for idx in game_indices], axis=0)

    # Get top similar games
    sim_scores = list(enumerate(combined_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filter out the input games themselves
    sim_scores = [score for score in sim_scores if score[0] not in game_indices]
    sim_scores = sim_scores[:200]

    item_indices = [i[0] for i in sim_scores]
    similarity_scores = {i[0]: i[1] for i in sim_scores}

    # Get recommendations
    recs = _df[['Name', 'Year Published', 'Min Players', 'Max Players', 'Play Time',
                'Rating Average', 'Complexity Average', 'Domains', 'Mechanics', 'ID']].iloc[item_indices].copy()

    # Add similarity score
    recs['Similarity'] = recs.index.map(similarity_scores)

    # Filter out expansions and similar titles
    all_exclusions = []
    for name in game_names_found:
        all_exclusions.extend(find_similar_titles(name))
    filtered_recs = recs[~recs['Name'].isin(all_exclusions)]

    # Apply user preference filters
    if player_count is not None:
        filtered_recs = filtered_recs[
            (filtered_recs['Min Players'] <= player_count) &
            (filtered_recs['Max Players'] >= player_count)
        ]

    if complexity_range is not None:
        filtered_recs = filtered_recs[
            (filtered_recs['Complexity Average'] >= complexity_range[0]) &
            (filtered_recs['Complexity Average'] <= complexity_range[1])
        ]

    if max_play_time is not None:
        filtered_recs = filtered_recs[filtered_recs['Play Time'] <= max_play_time]

    if min_rating is not None:
        filtered_recs = filtered_recs[filtered_recs['Rating Average'] >= min_rating]

    return filtered_recs.head(25).reset_index(drop=True), game_names_found, None

@st.cache_data
def explain_recommendation(rec_game_name, base_games, _df=df):
    """Explain why a game is recommended"""
    rec_game = _df[_df['Name'] == rec_game_name].iloc[0]

    explanations = []

    # Compare with each base game
    for base_game_name in base_games:
        base_game = _df[_df['Name'] == base_game_name].iloc[0]

        # Complexity similarity
        complexity_diff = abs(rec_game['Complexity Average'] - base_game['Complexity Average'])
        if complexity_diff < 0.5:
            explanations.append(f"✓ Similar complexity ({rec_game['Complexity Average']:.1f} vs {base_game['Complexity Average']:.1f})")

        # Shared mechanics
        rec_mechanics = set(rec_game['Mechanics'].split(', '))
        base_mechanics = set(base_game['Mechanics'].split(', '))
        shared = rec_mechanics & base_mechanics
        if shared and 'Not Specified' not in shared:
            mechanics_list = list(shared)[:3]  # Show top 3
            explanations.append(f"✓ Shared mechanics: {', '.join(mechanics_list)}")

        # Player count overlap
        if (rec_game['Min Players'] <= base_game['Max Players'] and
            rec_game['Max Players'] >= base_game['Min Players']):
            explanations.append(f"✓ Similar player count ({int(rec_game['Min Players'])}-{int(rec_game['Max Players'])} players)")

    # Rating
    if rec_game['Rating Average'] >= 7.5:
        explanations.append(f"✓ Highly rated ({rec_game['Rating Average']:.1f}/10)")

    return explanations[:4]  # Return top 4 reasons

def create_recommendation_html(recommendations_df, base_games, show_explanations=True):
    """Create HTML output with links and explanations"""
    html_output = ""
    for index, row in recommendations_df.iterrows():
        rec_url = f"https://boardgamegeek.com/boardgame/{row['ID']}/"
        match_percent = int(row.get('Similarity', 0) * 100)

        html_output += f"<div style='margin-bottom: 15px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>"
        html_output += f"<strong>{index + 1}. <a href='{rec_url}' target='_blank'>{row['Name']}</a></strong> "
        html_output += f"<span style='color: #0066cc;'>({match_percent}% match)</span><br>"
        html_output += f"<small>Year: {int(row['Year Published'])} | Rating: {row['Rating Average']:.1f}/10 | Complexity: {row['Complexity Average']:.1f}/5</small><br>"

        if show_explanations:
            explanations = explain_recommendation(row['Name'], base_games)
            if explanations:
                html_output += "<small style='color: #666;'>" + " | ".join(explanations) + "</small>"

        html_output += "</div>"

    return html_output

# Streamlit UI
st.title('🎲 Board Game Recommendation System')
st.markdown("*Discover your next favorite board game based on games you already love!*")

# Sidebar with filters
with st.sidebar:
    st.header("🎯 Preference Filters")
    st.markdown("*Filter recommendations to match your needs*")

    # Player count filter
    st.subheader("👥 Player Count")
    use_player_filter = st.checkbox("Filter by player count", value=False)
    if use_player_filter:
        player_count = st.slider("Number of players", 1, 10, 4)
    else:
        player_count = None

    # Complexity filter
    st.subheader("🧩 Complexity Level")
    use_complexity_filter = st.checkbox("Filter by complexity", value=False)
    if use_complexity_filter:
        complexity_range = st.slider(
            "Complexity range (1=Light, 5=Heavy)",
            1.0, 5.0, (1.0, 5.0), 0.1
        )
    else:
        complexity_range = None

    # Play time filter
    st.subheader("⏱️ Play Time")
    use_time_filter = st.checkbox("Filter by play time", value=False)
    if use_time_filter:
        max_play_time = st.slider("Maximum play time (minutes)", 15, 300, 120, 15)
    else:
        max_play_time = None

    # Rating filter
    st.subheader("⭐ Minimum Rating")
    use_rating_filter = st.checkbox("Filter by rating", value=False)
    if use_rating_filter:
        min_rating = st.slider("Minimum average rating", 1.0, 10.0, 7.0, 0.1)
    else:
        min_rating = None
     # Persistent toggle button for explanations
    if "show_explanations" not in st.session_state:
        st.session_state.show_explanations = False

    if st.button(
        "Hide explanations" if st.session_state.show_explanations else "Show why each game is recommended"
    ):
        st.session_state.show_explanations = not st.session_state.show_explanations

    show_explanations = st.session_state.show_explanations
    st.markdown("---")
    st.header("About")
    st.write(f"📊 Database: 20k+ board games")
    st.write("🎲 Games published in February 2021 or earlier")
    st.write("🔍 Pure content-based filtering")
    st.write("🎯 No popularity bias")

# Multi-game input option
st.markdown("### 🔍 Find Games Similar To...")
input_mode = st.radio("", ["Single game", "Multiple games (2-3)"], horizontal=True)

game_names = []

if input_mode == "Single game":
    game_query = st.text_input('Start typing a game name:', '', placeholder="e.g., Catan, Avalon, Ticket to Ride", key="game_search")

    # Show suggestions as user types
    if game_query and len(game_query) >= 2:
        suggestions = search_games(game_query)
        if suggestions:
            st.markdown("**Did you mean:**")
            selected_game = st.selectbox(
                "Select a game:",
                options=suggestions,
                key="game_selector"
            )
            game_names = [selected_game]
        else:
            st.warning(f"No games found matching '{game_query}'")
            game_names = [game_query] if game_query else []
    else:
        game_names = [game_query] if game_query else []
else:
    st.markdown("*Enter 2-3 games you like to get recommendations based on all of them:*")
    col1, col2, col3 = st.columns(3)

    with col1:
        game1 = st.text_input("Game 1:", "", key="multi_game1")
        if game1 and len(game1) >= 2:
            sugg1 = search_games(game1)
            if sugg1:
                game1 = st.selectbox("Select:", sugg1, key="select1")

    with col2:
        game2 = st.text_input("Game 2:", "", key="multi_game2")
        if game2 and len(game2) >= 2:
            sugg2 = search_games(game2)
            if sugg2:
                game2 = st.selectbox("Select:", sugg2, key="select2")

    with col3:
        game3 = st.text_input("Game 3 (optional):", "", key="multi_game3")
        if game3 and len(game3) >= 2:
            sugg3 = search_games(game3)
            if sugg3:
                game3 = st.selectbox("Select:", sugg3, key="select3")

    game_names = [g for g in [game1, game2, game3] if g]

search_button = st.button('🔍 Get Recommendations', type="primary", use_container_width=True, disabled=not any(game_names))

# Generate recommendations
if search_button and any(game_names):
    with st.spinner(f'🎲 Finding games similar to your selections...'):
        recommendations, base_games_found, error_msg = get_recommendations(
            game_names,
            player_count=player_count,
            complexity_range=complexity_range,
            max_play_time=max_play_time,
            min_rating=min_rating
        )

        if error_msg:
            st.warning(error_msg)
        elif recommendations is not None and not recommendations.empty:
            # Show which games are being used for recommendations
            if len(base_games_found) == 1:
                st.success(f'✅ Showing recommendations based on: **{base_games_found[0]}**')
            else:
                st.success(f'✅ Showing recommendations based on: **{", ".join(base_games_found)}**')

            # Show active filters
            active_filters = []
            if player_count:
                active_filters.append(f"👥 {player_count} players")
            if complexity_range:
                active_filters.append(f"🧩 Complexity {complexity_range[0]:.1f}-{complexity_range[1]:.1f}")
            if max_play_time:
                active_filters.append(f"⏱️ ≤{max_play_time} min")
            if min_rating:
                active_filters.append(f"⭐ ≥{min_rating:.1f} rating")

            if active_filters:
                st.info(f"**Active filters:** {' | '.join(active_filters)}")

            # Display game info cards for base games
            st.markdown("#### 📊 Base Game(s) Info:")
            cols = st.columns(len(base_games_found))
            for idx, base_game in enumerate(base_games_found):
                game_info = df[df['Name'] == base_game].iloc[0]
                with cols[idx]:
                    st.markdown(f"**{base_game}**")
                    st.caption(f"⭐ {game_info['Rating Average']:.1f} | 🧩 {game_info['Complexity Average']:.1f} | 👥 {int(game_info['Min Players'])}-{int(game_info['Max Players'])}")

            st.markdown("---")
            st.markdown(f"### 🌟 {len(recommendations)} Recommended Games")


            # Display as cards with explanations
            html_output = create_recommendation_html(recommendations, base_games_found, show_explanations)
            st.markdown(html_output, unsafe_allow_html=True)

            # Show detailed table
            with st.expander("📋 View Detailed Information Table"):
                display_df = recommendations[['Name', 'Year Published', 'Min Players', 'Max Players',
                                              'Play Time', 'Rating Average', 'Complexity Average']].copy()
                display_df.columns = ['Game', 'Year', 'Min Players', 'Max Players', 'Play Time (min)', 'Rating', 'Complexity']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.error('❌ No recommendations found matching your filters. Try adjusting your preferences.')
else:
    st.info('👆 Enter game name(s) above to get started!')

# Footer
st.markdown("---")
st.caption("Data source: BoardGameGeek | Built with Streamlit by Katie")
