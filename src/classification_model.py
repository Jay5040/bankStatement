import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

# File paths
INPUT_JSON = "../outputs/transactions_array.json"
OUTPUT_DIR = "../outputs/models"
RESULTS_DIR = "../outputs/results"

def load_data():
    """Load transaction data from the JSON file"""
    print("\n" + "="*70)
    print("CLASSIFICATION MODEL: Categorizing Transaction Descriptions")
    print("="*70 + "\n")
    
    # Check if file exists first
    if not os.path.exists(INPUT_JSON):
        print(f"❌ Error: {INPUT_JSON} not found!")
        print("Please run extract_template.py first")
        return None
    
    # Read the JSON file
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        transactions = json.load(f)
    
    df = pd.DataFrame(transactions)
    print(f"✅ Loaded {len(df)} transactions")
    
    # Show what categories we have
    print("\n📊 Category Distribution:")
    category_counts = df['spend_category'].value_counts()
    for category, count in category_counts.items():
        percentage = count/len(df)*100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    return df

def engineer_text_features(df):
    """Extract useful features from transaction descriptions"""
    print("\n📝 Engineering Text Features...")
    
    df = df.copy()
    
    # Make sure dates are datetime objects
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Clean up the text - lowercase and remove special characters
    df['description_clean'] = df['transaction_description'].str.lower()
    df['description_clean'] = df['description_clean'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['description_clean'] = df['description_clean'].str.replace(r'\s+', ' ', regex=True)
    df['description_clean'] = df['description_clean'].str.strip()
    
    # Create some text-based features
    df['description_length'] = df['transaction_description'].str.len()
    df['word_count'] = df['transaction_description'].str.split().str.len()
    df['has_numbers'] = df['transaction_description'].str.contains(r'\d').astype(int)
    
    # Average word length might be useful
    df['avg_word_length'] = df['description_clean'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x else 0
    )
    
    # Add date-related features
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['month'] = df['transaction_date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Log transform the amount for better distribution
    df['amount_log'] = np.log1p(df['amount'])
    
    # Turn card numbers into numeric codes
    le_card = LabelEncoder()
    df['card_encoded'] = le_card.fit_transform(df['card_number'])
    
    print(f"   ✓ Created text and numerical features")
    
    return df, le_card

def create_text_vectors(X_train_text, X_test_text):
    """Convert text descriptions into numbers using TF-IDF"""
    print("\n🔤 Vectorizing Text...")
    
    # TF-IDF is a way to represent text as numbers
    # It considers both word frequency and importance
    tfidf = TfidfVectorizer(
        max_features=100,  # Keep only top 100 words to avoid overfitting
        ngram_range=(1, 2),  # Use single words and pairs of words
        min_df=2,  # Word must appear at least twice
        max_df=0.8  # Ignore words that appear in more than 80% of transactions
    )
    
    # Fit on training data and transform both sets
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    print(f"   ✓ TF-IDF features: {X_train_tfidf.shape[1]}")
    print(f"   ✓ Sample features: {tfidf.get_feature_names_out()[:10].tolist()}")
    
    return X_train_tfidf, X_test_tfidf, tfidf

def combine_features(X_tfidf, X_numeric):
    """Merge text features with numeric features"""
    from scipy.sparse import hstack
    
    # Convert the numeric dataframe to array
    X_numeric_sparse = pd.DataFrame(X_numeric).to_numpy()
    
    # Stack them horizontally
    X_combined = hstack([X_tfidf, X_numeric_sparse])
    
    return X_combined

def train_models(X_train, X_test, y_train, y_test, class_names):
    """Train different classifiers and compare them"""
    print("\n🤖 Training Models...")
    
    # Define several models to try
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            C=1.0,  # Controls regularization strength
            random_state=42
        ),
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate different metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
        test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
        test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        
        # Use cross-validation to check stability
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        # Get detailed per-category performance
        labels = list(range(len(class_names)))
        report = classification_report(y_test, y_pred_test, 
                                       labels=labels,
                                       target_names=class_names, 
                                       output_dict=True,
                                       zero_division=0)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Store all the results
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_f1_weighted': test_f1_weighted,
            'test_f1_macro': test_f1_macro,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'cv_acc': cv_acc,
            'predictions': y_pred_test,
            'confusion_matrix': cm,
            'classification_report': report,
            'overfitting_score': train_acc - test_acc
        }
        
        # Print results
        print(f"      Train Accuracy: {train_acc:.4f}")
        print(f"      Test Accuracy:  {test_acc:.4f}")
        print(f"      Test F1 (weighted): {test_f1_weighted:.4f}")
        print(f"      Test F1 (macro): {test_f1_macro:.4f}")
        print(f"      CV Accuracy: {cv_acc:.4f}")
        print(f"      Overfitting: {results[name]['overfitting_score']:.4f}")
    
    # Pick the best model (balance accuracy and overfitting)
    best_model_name = max(results.keys(), 
                          key=lambda k: results[k]['test_acc'] - results[k]['overfitting_score']*0.5)
    
    print(f"\n   🏆 Best Model: {best_model_name}")
    
    return results, best_model_name

def plot_results(results, best_model_name, class_names):
    """Create charts to visualize model performance"""
    print("\n📈 Creating Visualizations...")
    
    # Make sure output directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create a big figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Classification Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    
    # Plot 1: Training vs Test Accuracy
    train_acc = [results[m]['train_acc'] for m in models]
    test_acc = [results[m]['test_acc'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_acc, width, label='Train Accuracy', color='skyblue')
    axes[0, 0].bar(x + width/2, test_acc, width, label='Test Accuracy', color='coral')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy: Train vs Test')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])
    
    # Plot 2: F1 Scores
    f1_weighted = [results[m]['test_f1_weighted'] for m in models]
    f1_macro = [results[m]['test_f1_macro'] for m in models]
    
    axes[0, 1].bar(x - width/2, f1_weighted, width, label='F1 Weighted', color='green')
    axes[0, 1].bar(x + width/2, f1_macro, width, label='F1 Macro', color='lightgreen')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Scores')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.1])
    
    # Plot 3: Overfitting Analysis
    overfitting = [results[m]['overfitting_score'] for m in models]
    # Color code: green = good, orange = ok, red = bad
    colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' for x in overfitting]
    axes[1, 0].bar(models, overfitting, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Overfitting Score (Train Acc - Test Acc)')
    axes[1, 0].set_title('Overfitting Analysis (Lower is Better)')
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', label='Acceptable (<0.1)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Precision and Recall
    precision = [results[m]['test_precision'] for m in models]
    recall = [results[m]['test_recall'] for m in models]
    
    axes[1, 1].bar(x - width/2, precision, width, label='Precision', color='purple')
    axes[1, 1].bar(x + width/2, recall, width, label='Recall', color='pink')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'classification_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: classification_comparison.png")
    plt.close()
    
    # Create confusion matrix for the best model
    cm = results[best_model_name]['confusion_matrix']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: confusion_matrix.png")
    plt.close()
    
    # Show performance for each category
    report = results[best_model_name]['classification_report']
    
    categories = [c for c in class_names if c in report]
    f1_scores = [report[c]['f1-score'] for c in categories]
    precision_scores = [report[c]['precision'] for c in categories]
    recall_scores = [report[c]['recall'] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision_scores, width, label='Precision', color='purple', alpha=0.8)
    ax.bar(x, recall_scores, width, label='Recall', color='orange', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='green', alpha=0.8)
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Category Performance - {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'category_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: category_performance.png")
    plt.close()

def save_results(results, best_model_name, class_names, tfidf, le_card, numeric_features):
    """Save the model and results to files"""
    print("\n💾 Saving Results...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert numpy types to regular Python types for JSON
    best_report = results[best_model_name]['classification_report']
    json_safe_report = {}
    for key, value in best_report.items():
        if isinstance(value, dict):
            json_safe_report[key] = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                     for k, v in value.items()}
        else:
            json_safe_report[key] = float(value) if isinstance(value, (np.floating, float)) else value
    
    # Create summary report
    report = {
        'best_model': best_model_name,
        'categories': class_names,
        'numeric_features': numeric_features,
        'tfidf_features': int(tfidf.max_features) if tfidf.max_features else 100,
        'model_performance': {
            name: {
                'test_accuracy': float(res['test_acc']),
                'test_f1_weighted': float(res['test_f1_weighted']),
                'test_f1_macro': float(res['test_f1_macro']),
                'test_precision': float(res['test_precision']),
                'test_recall': float(res['test_recall']),
                'cv_accuracy': float(res['cv_acc']),
                'overfitting_score': float(res['overfitting_score'])
            }
            for name, res in results.items()
        },
        'best_model_details': {
            'test_accuracy': float(results[best_model_name]['test_acc']),
            'test_f1_weighted': float(results[best_model_name]['test_f1_weighted']),
            'classification_report': json_safe_report,
            'interpretation': f"The model categorizes transactions with {results[best_model_name]['test_acc']*100:.2f}% accuracy"
        }
    }
    
    # Save as JSON
    output_path = os.path.join(OUTPUT_DIR, 'classification_results.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✓ Saved: classification_results.json")
    
    # Save the actual model using pickle
    import pickle
    model_data = {
        'model': results[best_model_name]['model'],
        'tfidf_vectorizer': tfidf,
        'label_encoder_card': le_card,
        'class_names': class_names,
        'numeric_features': numeric_features
    }
    
    output_path = os.path.join(OUTPUT_DIR, 'classification_model.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"   ✓ Saved: classification_model.pkl")

def main():
    # Step 1: Load the data
    df = load_data()
    if df is None:
        return
    
    # Step 2: Filter to just charges (not payments)
    df_charges = df[df['transaction_type'] == 'charge'].copy()
    print(f"\n🔍 Filtering to charges only: {len(df_charges)} transactions")
    
    # Step 3: Create features
    df_charges, le_card = engineer_text_features(df_charges)
    
    # Step 4: Prepare the data
    print("\n🔧 Preparing Data...")
    X_text = df_charges['description_clean']
    X_numeric = df_charges[['description_length', 'word_count', 'has_numbers', 
                            'avg_word_length', 'day_of_week', 'month', 
                            'is_weekend', 'amount_log', 'card_encoded']]
    y = df_charges['spend_category']
    
    # Convert category names to numbers
    le_category = LabelEncoder()
    y_encoded = le_category.fit_transform(y)
    class_names = le_category.classes_.tolist()
    
    print(f"   ✓ Text samples: {len(X_text)}")
    print(f"   ✓ Numeric features: {X_numeric.shape[1]}")
    print(f"   ✓ Categories: {len(class_names)}")
    
    # Check if we can use stratified split
    class_counts = pd.Series(y_encoded).value_counts()
    min_class_count = class_counts.min()
    
    # Step 5: Split into training and test sets
    print("\n✂️  Splitting Data...")
    if min_class_count >= 2:
        # Use stratified split to keep category proportions
        X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_text, X_numeric, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print(f"   ✓ Using stratified split")
    else:
        # Some categories are too rare for stratification
        print(f"   ⚠️  Warning: Some categories have only {min_class_count} sample(s)")
        print(f"   ✓ Using random split (stratification disabled)")
        X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_text, X_numeric, y_encoded, test_size=0.2, random_state=42
        )
    
    print(f"   ✓ Training set: {len(X_text_train)} samples")
    print(f"   ✓ Test set: {len(X_text_test)} samples")
    
    # Step 6: Convert text to numbers
    X_train_tfidf, X_test_tfidf, tfidf = create_text_vectors(X_text_train, X_text_test)
    
    # Step 7: Combine text and numeric features
    print("\n🔗 Combining Features...")
    X_train = combine_features(X_train_tfidf, X_num_train)
    X_test = combine_features(X_test_tfidf, X_num_test)
    print(f"   ✓ Total features: {X_train.shape[1]}")
    
    # Step 8: Train the models
    results, best_model_name = train_models(X_train, X_test, y_train, y_test, class_names)
    
    # Step 9: Create visualizations
    plot_results(results, best_model_name, class_names)
    
    # Step 10: Save everything
    numeric_features = X_numeric.columns.tolist()
    save_results(results, best_model_name, class_names, tfidf, le_card, numeric_features)
    
    # Step 11: Print final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test Accuracy: {results[best_model_name]['test_acc']*100:.2f}%")
    print(f"   Test F1 (weighted): {results[best_model_name]['test_f1_weighted']:.4f}")
    print(f"   Test F1 (macro): {results[best_model_name]['test_f1_macro']:.4f}")
    print(f"   Overfitting Score: {results[best_model_name]['overfitting_score']:.4f}")
    
    # Give feedback on overfitting
    overfit_score = results[best_model_name]['overfitting_score']
    if overfit_score < 0.1:
        print(f"\n✅ Model shows minimal overfitting (score < 0.1)")
    elif overfit_score < 0.2:
        print(f"\n⚠️  Model shows acceptable overfitting (score < 0.2)")
    else:
        print(f"\n❌ Model may be overfitting (score >= 0.2)")
    
    # Show top performing categories
    print(f"\n📊 Top 3 Categories by Performance:")
    report = results[best_model_name]['classification_report']
    categories_f1 = [(cat, report[cat]['f1-score']) for cat in class_names if cat in report]
    categories_f1.sort(key=lambda x: x[1], reverse=True)
    for i, (cat, f1) in enumerate(categories_f1[:3], 1):
        print(f"   {i}. {cat}: F1 = {f1:.4f}")
    
    print(f"\n📊 Visualizations saved to: {RESULTS_DIR}/")
    print(f"💾 Model saved to: {OUTPUT_DIR}/")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()