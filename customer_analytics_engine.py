#!/usr/bin/env python3
"""
🔬 Advanced Customer Analytics Engine
AI-powered segmentation, behavioral forecasting, and customer lifetime value prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced ML Libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy import stats
import networkx as nx

# Interactive Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

class AdvancedCustomerAnalytics:
    """
    🧠 AI-Powered Customer Analytics Engine
    Features:
    - Advanced Customer Segmentation using ML
    - Behavioral Pattern Recognition
    - Customer Lifetime Value Prediction
    - Churn Risk Forecasting
    - Purchase Behavior Analysis
    - Network Analysis
    - Real-time Anomaly Detection
    """
    
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.visualizations = {}
        
    def generate_synthetic_data(self, n_customers=10000):
        """
        Generate sophisticated synthetic customer data with realistic patterns
        """
        print("🎲 Generating advanced synthetic customer data...")
        
        np.random.seed(42)
        
        # Customer demographics
        ages = np.random.normal(35, 10, n_customers).astype(int)
        ages = np.clip(ages, 18, 80)
        
        genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04])
        
        locations = np.random.choice(['North America', 'Europe', 'Asia', 'South America', 'Australia'], 
                                   n_customers, p=[0.4, 0.3, 0.2, 0.08, 0.02])
        
        # Behavioral patterns
        income_levels = np.random.lognormal(10.5, 0.8, n_customers)
        
        # Create correlated features (realistic patterns)
        spending_power = income_levels * np.random.normal(1, 0.2, n_customers)
        purchase_frequency = np.random.poisson(3, n_customers) + (spending_power / 10000).astype(int)
        
        # Customer engagement metrics
        days_since_last_purchase = np.random.exponential(30, n_customers)
        total_visits = np.random.negative_binomial(5, 0.3, n_customers) + 1
        session_duration = np.random.gamma(4, 2, n_customers)
        
        # Product preferences (multi-category)
        categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books', 'Beauty']
        preferred_category = np.random.choice(categories, n_customers)
        
        # Purchase history (time-series like data)
        total_spent = np.random.gamma(2, 100, n_customers) * (spending_power / 10000)
        avg_order_value = total_spent / np.maximum(purchase_frequency, 1)
        
        # Customer satisfaction scores
        satisfaction_scores = np.random.beta(2, 2, n_customers) * 10
        
        # Social influence metrics
        social_connections = np.random.poisson(15, n_customers)
        influencer_score = np.random.exponential(0.5, n_customers)
        
        # Device and platform preferences
        devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_customers, p=[0.6, 0.3, 0.1])
        platforms = np.random.choice(['iOS', 'Android', 'Web'], n_customers, p=[0.45, 0.45, 0.1])
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
            'age': ages,
            'gender': genders,
            'location': locations,
            'income_level': income_levels,
            'spending_power': spending_power,
            'purchase_frequency': purchase_frequency,
            'days_since_last_purchase': days_since_last_purchase,
            'total_visits': total_visits,
            'session_duration': session_duration,
            'preferred_category': preferred_category,
            'total_spent': total_spent,
            'avg_order_value': avg_order_value,
            'satisfaction_score': satisfaction_scores,
            'social_connections': social_connections,
            'influencer_score': influencer_score,
            'primary_device': devices,
            'primary_platform': platforms,
            'last_purchase_date': [datetime.now() - timedelta(days=int(x)) for x in days_since_last_purchase]
        })
        
        # Add some realistic correlations
        self.df['total_spent'] = self.df['total_spent'] * (1 + self.df['satisfaction_score'] / 20)
        self.df['purchase_frequency'] = np.clip(self.df['purchase_frequency'] + (self.df['age'] - 35) // 10, 1, 20)
        
        print(f"✅ Generated {len(self.df)} customer records with 18 features")
        return self.df
    
    def advanced_segmentation(self, n_clusters=6):
        """
        Advanced customer segmentation using multiple ML techniques
        """
        print("\n🎯 Performing advanced customer segmentation...")
        
        # Prepare features for clustering
        features_for_clustering = [
            'age', 'income_level', 'spending_power', 'purchase_frequency',
            'total_spent', 'avg_order_value', 'satisfaction_score',
            'social_connections', 'influencer_score', 'session_duration'
        ]
        
        X = self.df[features_for_clustering].copy()
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Method 1: K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # Method 2: DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # Evaluate clustering quality
        silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Add results to dataframe
        self.df['segment_kmeans'] = kmeans_labels
        self.df['segment_dbscan'] = dbscan_labels
        self.df['pca_1'] = X_pca[:, 0]
        self.df['pca_2'] = X_pca[:, 1]
        
        # Analyze segment characteristics
        segment_profiles = self._analyze_segments(kmeans_labels, features_for_clustering)
        
        self.results['segmentation'] = {
            'kmeans_labels': kmeans_labels,
            'dbscan_labels': dbscan_labels,
            'silhouette_score': silhouette_kmeans,
            'segment_profiles': segment_profiles,
            'pca_explained_variance': pca.explained_variance_ratio_
        }
        
        print(f"✅ Segmentation completed with silhouette score: {silhouette_kmeans:.3f}")
        return segment_profiles
    
    def _analyze_segments(self, labels, features):
        """Analyze characteristics of each segment"""
        segment_analysis = {}
        
        for segment in np.unique(labels):
            segment_data = self.df[self.df['segment_kmeans'] == segment]
            
            profile = {
                'size': len(segment_data),
                'demographics': {
                    'avg_age': segment_data['age'].mean(),
                    'gender_distribution': segment_data['gender'].value_counts().to_dict(),
                    'location_distribution': segment_data['location'].value_counts().to_dict()
                },
                'behavioral': {
                    'avg_income': segment_data['income_level'].mean(),
                    'avg_spending': segment_data['total_spent'].mean(),
                    'avg_frequency': segment_data['purchase_frequency'].mean(),
                    'avg_satisfaction': segment_data['satisfaction_score'].mean()
                },
                'preferences': {
                    'top_category': segment_data['preferred_category'].mode().iloc[0] if not segment_data['preferred_category'].empty else 'N/A',
                    'top_device': segment_data['primary_device'].mode().iloc[0] if not segment_data['primary_device'].empty else 'N/A'
                }
            }
            segment_analysis[f'Segment_{segment}'] = profile
        
        return segment_analysis
    
    def predict_customer_lifetime_value(self):
        """
        Predict Customer Lifetime Value using advanced ML models
        """
        print("\n💰 Predicting Customer Lifetime Value...")
        
        # Create CLV target variable (simulated)
        # In real scenarios, this would be based on historical data
        self.df['predicted_clv'] = (
            self.df['total_spent'] * 
            (1 + self.df['satisfaction_score'] / 10) * 
            (1 + np.log1p(self.df['social_connections']) / 10) *
            (1 - self.df['days_since_last_purchase'] / 365)
        )
        
        # Features for CLV prediction
        clv_features = [
            'age', 'income_level', 'purchase_frequency', 'total_spent',
            'avg_order_value', 'satisfaction_score', 'social_connections',
            'influencer_score', 'session_duration', 'days_since_last_purchase'
        ]
        
        X = self.df[clv_features].fillna(self.df[clv_features].mean())
        y = self.df['predicted_clv']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Convert to classification for demonstration
        y_train_class = pd.cut(y_train, bins=5, labels=[1, 2, 3, 4, 5])
        y_test_class = pd.cut(y_test, bins=5, labels=[1, 2, 3, 4, 5])
        
        model_performance = {}
        
        for name, model in models.items():
            if name == 'RandomForest':
                model.fit(X_train, y_train_class)
                y_pred = model.predict(X_test)
                accuracy = (y_pred == y_test_class).mean()
                model_performance[name] = accuracy
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                model_performance[name] = mse
        
        # Store best model
        best_model_name = min(model_performance, key=model_performance.get)
        self.models['clv'] = models[best_model_name]
        
        self.results['clv_prediction'] = {
            'model_performance': model_performance,
            'best_model': best_model_name,
            'feature_importance': dict(zip(clv_features, self.models['clv'].feature_importances_))
        }
        
        print(f"✅ CLV Prediction completed. Best model: {best_model_name}")
        return model_performance
    
    def churn_risk_analysis(self):
        """
        Advanced churn risk prediction and analysis
        """
        print("\n⚠️  Analyzing customer churn risk...")
        
        # Create churn indicator (simulated based on behavior patterns)
        # Customers with high days_since_last_purchase and low engagement are at risk
        self.df['churn_risk_score'] = (
            (self.df['days_since_last_purchase'] / 30) * 0.4 +  # Recency factor
            (1 - (self.df['purchase_frequency'] / self.df['purchase_frequency'].max())) * 0.3 +  # Frequency factor
            (1 - (self.df['session_duration'] / self.df['session_duration'].max())) * 0.2 +  # Engagement factor
            (1 - (self.df['satisfaction_score'] / 10)) * 0.1  # Satisfaction factor
        )
        
        # Classify churn risk levels
        self.df['churn_risk_level'] = pd.cut(
            self.df['churn_risk_score'], 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Analyze risk factors by segment
        churn_analysis = self.df.groupby(['segment_kmeans', 'churn_risk_level']).agg({
            'customer_id': 'count',
            'days_since_last_purchase': 'mean',
            'satisfaction_score': 'mean',
            'total_spent': 'mean'
        }).reset_index()
        
        self.results['churn_analysis'] = {
            'overall_churn_risk': self.df['churn_risk_level'].value_counts().to_dict(),
            'segment_churn_analysis': churn_analysis,
            'high_risk_customers': self.df[self.df['churn_risk_level'] == 'High'].shape[0]
        }
        
        print(f"🚨 Identified {self.results['churn_analysis']['high_risk_customers']} high-risk customers")
        return churn_analysis
    
    def behavioral_pattern_analysis(self):
        """
        Advanced behavioral pattern recognition and analysis
        """
        print("\n🔍 Analyzing behavioral patterns...")
        
        # Purchase pattern analysis
        purchase_patterns = self.df.groupby('preferred_category').agg({
            'total_spent': ['mean', 'sum', 'count'],
            'purchase_frequency': 'mean',
            'avg_order_value': 'mean',
            'satisfaction_score': 'mean'
        }).round(2)
        
        # Customer engagement analysis
        engagement_metrics = self.df.groupby('segment_kmeans').agg({
            'session_duration': ['mean', 'std'],
            'total_visits': ['mean', 'sum'],
            'days_since_last_purchase': ['mean', 'min', 'max']
        }).round(2)
        
        # Correlation analysis
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_columns].corr()
        
        # Behavioral clustering based on engagement
        engagement_features = ['session_duration', 'total_visits', 'purchase_frequency', 'social_connections']
        X_engagement = self.df[engagement_features].fillna(self.df[engagement_features].mean())
        X_engagement_scaled = StandardScaler().fit_transform(X_engagement)
        
        engagement_clusters = KMeans(n_clusters=4, random_state=42).fit_predict(X_engagement_scaled)
        self.df['engagement_cluster'] = engagement_clusters
        
        self.results['behavioral_analysis'] = {
            'purchase_patterns': purchase_patterns,
            'engagement_metrics': engagement_metrics,
            'correlation_matrix': correlation_matrix,
            'engagement_clusters': engagement_clusters
        }
        
        print("✅ Behavioral pattern analysis completed")
        return purchase_patterns
    
    def network_analysis(self):
        """
        Social network analysis for customer influence mapping
        """
        print("\n🕸️  Performing network analysis...")
        
        # Create a synthetic social network (in real scenario, this would be actual data)
        n_customers = min(200, len(self.df))  # Limit for performance
        G = nx.Graph()
        
        # Add nodes (customers)
        sample_customers = self.df.head(n_customers)
        for _, customer in sample_customers.iterrows():
            G.add_node(customer['customer_id'], 
                      influence=customer['influencer_score'],
                      segment=customer['segment_kmeans'])
        
        # Add edges (connections) based on similarity and influence
        for i in range(n_customers):
            for j in range(i+1, min(i+50, n_customers)):
                cust1 = sample_customers.iloc[i]
                cust2 = sample_customers.iloc[j]
                
                # Connection probability based on similarity and influence
                similarity = 1 / (1 + abs(cust1['age'] - cust2['age']) / 10 +
                                abs(cust1['income_level'] - cust2['income_level']) / 10000)
                
                influence_factor = (cust1['influencer_score'] + cust2['influencer_score']) / 2
                
                if np.random.random() < similarity * influence_factor * 0.1:
                    G.add_edge(cust1['customer_id'], cust2['customer_id'], weight=similarity)
        
        # Calculate network metrics
        if G.number_of_edges() > 0:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            pagerank = nx.pagerank(G)
            
            # Add metrics to sample customers
            for customer_id in sample_customers['customer_id']:
                if customer_id in degree_centrality:
                    idx = self.df[self.df['customer_id'] == customer_id].index
                    if not idx.empty:
                        self.df.loc[idx, 'network_centrality'] = degree_centrality[customer_id]
                        self.df.loc[idx, 'network_influence'] = pagerank.get(customer_id, 0)
        
        network_metrics = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G)
        }
        
        self.results['network_analysis'] = network_metrics
        print(f"✅ Network analysis completed: {network_metrics['nodes']} nodes, {network_metrics['edges']} edges")
        return network_metrics
    
    def create_advanced_visualizations(self):
        """
        Create interactive and advanced visualizations
        """
        print("\n📊 Creating advanced visualizations...")
        
        # 1. Segmentation Visualization
        fig_segment = px.scatter(self.df, x='pca_1', y='pca_2', color='segment_kmeans',
                               hover_data=['age', 'total_spent', 'satisfaction_score'],
                               title='Customer Segmentation (PCA Projection)')
        
        # 2. CLV Distribution by Segment
        fig_clv = px.box(self.df, x='segment_kmeans', y='predicted_clv',
                        color='segment_kmeans', title='Customer Lifetime Value by Segment')
        
        # 3. Churn Risk Analysis
        churn_counts = self.df['churn_risk_level'].value_counts()
        fig_churn = px.pie(values=churn_counts.values, names=churn_counts.index,
                          title='Customer Churn Risk Distribution')
        
        # 4. Behavioral Heatmap
        corr_matrix = self.df[['age', 'income_level', 'total_spent', 'purchase_frequency', 
                             'satisfaction_score', 'session_duration']].corr()
        fig_heatmap = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            annotation_text=corr_matrix.round(2).values,
            colorscale='Viridis'
        )
        fig_heatmap.update_layout(title='Feature Correlation Heatmap')
        
        self.visualizations = {
            'segmentation': fig_segment,
            'clv_distribution': fig_clv,
            'churn_risk': fig_churn,
            'correlation_heatmap': fig_heatmap
        }
        
        print("✅ Advanced visualizations created")
        return self.visualizations
    
    def generate_insights_report(self):
        """
        Generate comprehensive insights and recommendations
        """
        print("\n📈 Generating insights report...")
        
        insights = {
            'key_findings': [],
            'segmentation_insights': [],
            'risk_alerts': [],
            'growth_opportunities': [],
            'strategic_recommendations': []
        }
        
        # Key findings
        total_customers = len(self.df)
        high_value_customers = len(self.df[self.df['predicted_clv'] > self.df['predicted_clv'].quantile(0.75)])
        high_risk_customers = len(self.df[self.df['churn_risk_level'] == 'High'])
        
        insights['key_findings'].extend([
            f"Total customer base: {total_customers:,}",
            f"High-value customers identified: {high_value_customers:,}",
            f"High churn-risk customers: {high_risk_customers:,}",
            f"Average customer lifetime value: ${self.df['predicted_clv'].mean():.2f}",
            f"Overall satisfaction score: {self.df['satisfaction_score'].mean():.1f}/10"
        ])
        
        # Segmentation insights
        if 'segmentation' in self.results:
            segments = self.results['segmentation']['segment_profiles']
            for segment, profile in segments.items():
                insights['segmentation_insights'].append(
                    f"{segment}: {profile['size']} customers, "
                    f"Avg spending: ${profile['behavioral']['avg_spending']:.2f}, "
                    f"Satisfaction: {profile['behavioral']['avg_satisfaction']:.1f}/10"
                )
        
        # Risk alerts
        if 'churn_analysis' in self.results:
            insights['risk_alerts'].extend([
                f"🚨 {high_risk_customers} customers at high risk of churn",
                f"⚠️  {len(self.df[self.df['days_since_last_purchase'] > 90])} customers inactive for 90+ days"
            ])
        
        # Growth opportunities
        high_potential_segments = self.df.groupby('segment_kmeans')['predicted_clv'].mean().nlargest(2)
        for segment, clv in high_potential_segments.items():
            insights['growth_opportunities'].append(
                f"Segment {segment} shows high CLV potential (${clv:.2f})"
            )
        
        # Strategic recommendations
        insights['strategic_recommendations'].extend([
            "🎯 Focus retention efforts on high-risk segments identified in the analysis",
            "💡 Develop personalized marketing campaigns for each customer segment",
            "📱 Optimize mobile experience for segments with high mobile usage",
            "🤝 Leverage influencer networks for customer acquisition",
            "📊 Implement real-time monitoring of customer engagement metrics"
        ])
        
        self.results['insights_report'] = insights
        
        print("✅ Comprehensive insights report generated")
        return insights
    
    def save_analysis_results(self):
        """
        Save all analysis results and visualizations
        """
        print("\n💾 Saving analysis results...")
        
        # Save dataframe
        self.df.to_csv('sample_data/customer_data.csv', index=False)
        
        # Save results summary
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_customers': len(self.df),
            'segments_identified': len(self.df['segment_kmeans'].unique()),
            'analysis_summary': self.results
        }
        
        import json
        with open('sample_data/analysis_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Save visualizations as HTML
        for viz_name, fig in self.visualizations.items():
            fig.write_html(f'sample_data/{viz_name}_visualization.html')
        
        print("✅ All results saved to sample_data/ directory")
    
    def run_complete_analysis(self):
        """
        Run the complete advanced analytics pipeline
        """
        print("🚀 STARTING ADVANCED CUSTOMER ANALYTICS PIPELINE")
        print("=" * 60)
        
        # Step 1: Generate or load data
        self.generate_synthetic_data(5000)
        
        # Step 2: Advanced segmentation
        self.advanced_segmentation()
        
        # Step 3: CLV prediction
        self.predict_customer_lifetime_value()
        
        # Step 4: Churn risk analysis
        self.churn_risk_analysis()
        
        # Step 5: Behavioral analysis
        self.behavioral_pattern_analysis()
        
        # Step 6: Network analysis
        self.network_analysis()
        
        # Step 7: Visualizations
        self.create_advanced_visualizations()
        
        # Step 8: Insights report
        insights = self.generate_insights_report()
        
        # Step 9: Save results
        self.save_analysis_results()
        
        print("\n🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Print key insights
        print("\n📊 KEY INSIGHTS SUMMARY:")
        print("-" * 25)
        for finding in insights['key_findings'][:3]:
            print(f"• {finding}")
        
        print("\n🎯 TOP RECOMMENDATIONS:")
        print("-" * 25)
        for recommendation in insights['strategic_recommendations'][:3]:
            print(f"• {recommendation}")
        
        return self.results

def main():
    """
    Main function to demonstrate the advanced analytics engine
    """
    # Initialize the analytics engine
    analytics_engine = AdvancedCustomerAnalytics()
    
    # Run complete analysis
    results = analytics_engine.run_complete_analysis()
    
    print(f"\n📁 Results saved in 'sample_data/' directory")
    print("🔍 Open the HTML files in your browser to view interactive visualizations")

if __name__ == "__main__":
    main()
