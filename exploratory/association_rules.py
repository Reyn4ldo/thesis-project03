"""
Association rule mining for co-resistance pattern discovery.

Uses Apriori and FP-growth algorithms to find frequent itemsets
and association rules among antibiotic resistances.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')


class AssociationRuleMiner:
    """
    Mine association rules for co-resistance patterns.
    
    Discovers frequently co-occurring antibiotic resistances and
    generates rules with support, confidence, and lift metrics.
    """
    
    def __init__(self, min_support=0.1, min_confidence=0.5, min_lift=1.0):
        """
        Initialize miner.
        
        Parameters
        ----------
        min_support : float, default=0.1
            Minimum support threshold (0-1)
        min_confidence : float, default=0.5
            Minimum confidence threshold (0-1)
        min_lift : float, default=1.0
            Minimum lift threshold
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        
        self.frequent_itemsets = None
        self.rules = None
    
    def prepare_transactions(self, df):
        """
        Prepare transaction data from resistance profiles.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with binary resistance columns
            
        Returns
        -------
        pd.DataFrame
            Binary transaction dataframe suitable for apriori
        """
        # Get resistance columns
        resistance_cols = [col for col in df.columns 
                          if '_resistant' in col and 'total' not in col]
        
        # Create binary dataframe
        transactions = df[resistance_cols].copy()
        
        # Ensure binary values
        transactions = transactions.astype(bool)
        
        # Rename columns to antibiotic names (remove _resistant suffix)
        transactions.columns = [col.replace('_resistant', '') for col in transactions.columns]
        
        print(f"Prepared {len(transactions)} transactions with {len(transactions.columns)} antibiotics")
        
        return transactions
    
    def mine_apriori(self, transactions, min_support=None):
        """
        Mine frequent itemsets using Apriori algorithm.
        
        Parameters
        ----------
        transactions : pd.DataFrame
            Binary transaction dataframe
        min_support : float, optional
            Minimum support (overrides default)
            
        Returns
        -------
        pd.DataFrame
            Frequent itemsets with support values
        """
        if min_support is None:
            min_support = self.min_support
        
        print(f"\nMining frequent itemsets (Apriori) with min_support={min_support}...")
        
        self.frequent_itemsets = apriori(
            transactions,
            min_support=min_support,
            use_colnames=True
        )
        
        # Sort by support
        self.frequent_itemsets = self.frequent_itemsets.sort_values(
            'support', ascending=False
        ).reset_index(drop=True)
        
        print(f"  Found {len(self.frequent_itemsets)} frequent itemsets")
        
        return self.frequent_itemsets
    
    def mine_fpgrowth(self, transactions, min_support=None):
        """
        Mine frequent itemsets using FP-growth algorithm.
        
        Parameters
        ----------
        transactions : pd.DataFrame
            Binary transaction dataframe
        min_support : float, optional
            Minimum support (overrides default)
            
        Returns
        -------
        pd.DataFrame
            Frequent itemsets with support values
        """
        if min_support is None:
            min_support = self.min_support
        
        print(f"\nMining frequent itemsets (FP-growth) with min_support={min_support}...")
        
        self.frequent_itemsets = fpgrowth(
            transactions,
            min_support=min_support,
            use_colnames=True
        )
        
        # Sort by support
        self.frequent_itemsets = self.frequent_itemsets.sort_values(
            'support', ascending=False
        ).reset_index(drop=True)
        
        print(f"  Found {len(self.frequent_itemsets)} frequent itemsets")
        
        return self.frequent_itemsets
    
    def generate_rules(self, metric='lift', min_threshold=None):
        """
        Generate association rules from frequent itemsets.
        
        Parameters
        ----------
        metric : str, default='lift'
            Metric to filter by ('support', 'confidence', 'lift')
        min_threshold : float, optional
            Minimum threshold for the metric
            
        Returns
        -------
        pd.DataFrame
            Association rules with metrics
        """
        if self.frequent_itemsets is None:
            raise ValueError("Must mine frequent itemsets first")
        
        if min_threshold is None:
            if metric == 'confidence':
                min_threshold = self.min_confidence
            elif metric == 'lift':
                min_threshold = self.min_lift
            else:
                min_threshold = self.min_support
        
        print(f"\nGenerating association rules (min_{metric}={min_threshold})...")
        
        self.rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold
        )
        
        # Add additional filtering
        self.rules = self.rules[
            (self.rules['confidence'] >= self.min_confidence) &
            (self.rules['lift'] >= self.min_lift)
        ]
        
        # Sort by lift (most interesting rules first)
        self.rules = self.rules.sort_values('lift', ascending=False).reset_index(drop=True)
        
        print(f"  Generated {len(self.rules)} association rules")
        
        return self.rules
    
    def get_top_rules(self, n=20, sort_by='lift'):
        """
        Get top N association rules.
        
        Parameters
        ----------
        n : int, default=20
            Number of rules to return
        sort_by : str, default='lift'
            Metric to sort by
            
        Returns
        -------
        pd.DataFrame
            Top N rules
        """
        if self.rules is None:
            raise ValueError("Must generate rules first")
        
        return self.rules.nlargest(n, sort_by)
    
    def get_rules_for_antibiotic(self, antibiotic):
        """
        Get rules involving a specific antibiotic.
        
        Parameters
        ----------
        antibiotic : str
            Antibiotic name
            
        Returns
        -------
        pd.DataFrame
            Rules involving the antibiotic
        """
        if self.rules is None:
            raise ValueError("Must generate rules first")
        
        # Find rules where antibiotic appears in antecedent or consequent
        mask = self.rules['antecedents'].apply(lambda x: antibiotic in x) | \
               self.rules['consequents'].apply(lambda x: antibiotic in x)
        
        return self.rules[mask].sort_values('lift', ascending=False)
    
    def format_rule(self, row):
        """
        Format a rule for display.
        
        Parameters
        ----------
        row : pd.Series
            Rule row from rules dataframe
            
        Returns
        -------
        str
            Formatted rule string
        """
        antecedents = ', '.join(sorted(row['antecedents']))
        consequents = ', '.join(sorted(row['consequents']))
        
        return (f"{antecedents} -> {consequents} "
                f"[support={row['support']:.3f}, "
                f"confidence={row['confidence']:.3f}, "
                f"lift={row['lift']:.3f}]")
    
    def print_top_rules(self, n=10, sort_by='lift'):
        """
        Print top N rules in human-readable format.
        
        Parameters
        ----------
        n : int, default=10
            Number of rules to print
        sort_by : str, default='lift'
            Metric to sort by
        """
        top_rules = self.get_top_rules(n, sort_by)
        
        print(f"\nTop {n} Association Rules (sorted by {sort_by}):")
        print("="*80)
        
        for idx, row in top_rules.iterrows():
            print(f"\n{idx+1}. {self.format_rule(row)}")
    
    def get_co_resistance_matrix(self, transactions):
        """
        Generate co-resistance frequency matrix.
        
        Parameters
        ----------
        transactions : pd.DataFrame
            Binary transaction dataframe
            
        Returns
        -------
        pd.DataFrame
            Co-resistance frequency matrix
        """
        antibiotics = transactions.columns
        n_antibiotics = len(antibiotics)
        
        # Initialize matrix
        co_resistance = pd.DataFrame(
            0, index=antibiotics, columns=antibiotics, dtype=float
        )
        
        # Calculate co-resistance frequencies
        for i, ab1 in enumerate(antibiotics):
            for j, ab2 in enumerate(antibiotics):
                if i <= j:
                    # Both resistant
                    both_resistant = (transactions[ab1] & transactions[ab2]).sum()
                    co_resistance.loc[ab1, ab2] = both_resistant / len(transactions)
                    co_resistance.loc[ab2, ab1] = co_resistance.loc[ab1, ab2]
        
        return co_resistance
    
    def analyze_itemset_sizes(self):
        """
        Analyze distribution of itemset sizes.
        
        Returns
        -------
        dict
            Statistics about itemset sizes
        """
        if self.frequent_itemsets is None:
            raise ValueError("Must mine frequent itemsets first")
        
        itemset_sizes = self.frequent_itemsets['itemsets'].apply(len)
        
        stats = {
            'min_size': itemset_sizes.min(),
            'max_size': itemset_sizes.max(),
            'mean_size': itemset_sizes.mean(),
            'size_distribution': itemset_sizes.value_counts().sort_index().to_dict()
        }
        
        return stats
    
    def get_summary_statistics(self):
        """
        Get summary statistics of mining results.
        
        Returns
        -------
        dict
            Summary statistics
        """
        summary = {
            'n_frequent_itemsets': len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
            'n_rules': len(self.rules) if self.rules is not None else 0
        }
        
        if self.frequent_itemsets is not None:
            summary['itemset_stats'] = self.analyze_itemset_sizes()
        
        if self.rules is not None:
            summary['rule_stats'] = {
                'mean_support': self.rules['support'].mean(),
                'mean_confidence': self.rules['confidence'].mean(),
                'mean_lift': self.rules['lift'].mean(),
                'max_lift': self.rules['lift'].max()
            }
        
        return summary
