"""
Advanced Sentiment Analysis for Customer Service Calls (UI-agnostic).

This module provides sentiment analysis capabilities using multiple models:
- VADER sentiment analyzer
- BERT-based sentiment classifier
- Emotion detection
- spaCy NLP analysis
"""

import torch
import numpy as np
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from typing import Dict, List
import logging

from config import Config

logger = logging.getLogger(__name__)


class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with multiple models and emotion detection."""

    def __init__(self):
        """Initialize sentiment analysis models."""
        self.device = Config.DEVICE
        self._load_models()

    def _load_models(self) -> None:
        """Load all sentiment analysis models."""
        try:
            # VADER sentiment analyzer (always works)
            self.vader_analyzer = SentimentIntensityAnalyzer()

            # Simple sentiment classifier (more reliable)
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" else -1
            )

            # Simple emotion classifier - use a proper emotion model
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if self.device == "cuda" else -1
            )

            # Load spaCy model for advanced NLP
            self.nlp = spacy.load("en_core_web_sm")

            logger.info("All sentiment analysis models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading sentiment models: {e}")
            # Fallback to VADER only
            self.sentiment_classifier = None
            self.emotion_classifier = None
            self.nlp = None
            logger.info("Using VADER-only fallback mode")

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Comprehensive sentiment analysis of text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with sentiment analysis results.
        """
        if not text or not text.strip():
            return self._empty_sentiment_result()

        try:
            # VADER sentiment scores (always available)
            vader_scores = self.vader_analyzer.polarity_scores(text)

            # BERT sentiment analysis (if available)
            if self.sentiment_classifier:
                bert_result = self.sentiment_classifier(text)[0]
            else:
                bert_result = {'label': 'NEUTRAL', 'score': 0.5}

            # Emotion detection (if available)
            if self.emotion_classifier:
                emotion_result = self.emotion_classifier(text)[0]
            else:
                emotion_result = {'label': 'neutral', 'score': 1.0}

            # Advanced NLP analysis (if available)
            if self.nlp:
                doc = self.nlp(text)
                nlp_features = self._extract_nlp_features(doc)
            else:
                nlp_features = {'word_count': len(text.split())}

            # Calculate overall sentiment score
            overall_sentiment = self._calculate_overall_sentiment(
                vader_scores, bert_result, emotion_result
            )

            return {
                'text': text,
                'vader_scores': vader_scores,
                'bert_sentiment': bert_result,
                'emotion': emotion_result,
                'nlp_features': nlp_features,
                'overall_sentiment': overall_sentiment,
                'timestamp': None  # Will be set by caller
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._empty_sentiment_result()

    def analyze_conversation_sentiment(self, segments: List[Dict]) -> Dict:
        """
        Analyze sentiment for entire conversation.

        Args:
            segments: List of conversation segments.

        Returns:
            Dictionary with conversation-level sentiment analysis.
        """
        speaker_sentiments = {}

        for segment in segments:
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '')
            timestamp = segment.get('start', 0)

            if text.strip():
                sentiment_result = self.analyze_sentiment(text)
                sentiment_result['timestamp'] = timestamp

                if speaker not in speaker_sentiments:
                    speaker_sentiments[speaker] = []
                speaker_sentiments[speaker].append(sentiment_result)

        # Calculate conversation-level metrics
        conversation_metrics = self._calculate_conversation_metrics(speaker_sentiments)

        return {
            'speaker_sentiments': speaker_sentiments,
            'conversation_metrics': conversation_metrics,
            'total_segments': len(segments)
        }

    def _extract_nlp_features(self, doc) -> Dict:
        """Extract advanced NLP features from spaCy document."""
        features = {
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents)),
            'avg_word_length': np.mean([len(token.text) for token in doc]),
            'exclamation_count': sum(1 for token in doc if token.text == '!'),
            'question_count': sum(1 for token in doc if token.text == '?'),
            'uppercase_ratio': sum(1 for token in doc if token.is_upper) / len(doc),
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'pos_tags': [token.pos_ for token in doc],
            'negation_words': sum(1 for token in doc if token.dep_ == 'neg')
        }
        return features

    def _calculate_overall_sentiment(
        self,
        vader_scores: Dict,
        bert_result: Dict,
        emotion_result: Dict
    ) -> Dict:
        """Calculate overall sentiment score from multiple models - RELIABLE VERSION."""
        # VADER compound score (-1 to 1) - PRIMARY SOURCE (most reliable)
        vader_score = vader_scores['compound']

        # VADER individual scores for better understanding
        vader_pos = vader_scores['pos']
        vader_neg = vader_scores['neg']
        vader_neu = vader_scores['neu']

        # BERT score (convert to -1 to 1 scale) - use as secondary check
        if bert_result['label'] == 'POSITIVE':
            bert_score = bert_result['score'] * 0.5  # Scale down BERT influence
        elif bert_result['label'] == 'NEGATIVE':
            bert_score = -bert_result['score'] * 0.5
        else:
            bert_score = 0.0

        # Emotion score (convert to sentiment) - minimal influence
        emotion_scores = {
            'joy': 0.6, 'love': 0.5, 'surprise': 0.2, 'neutral': 0.0,
            'sadness': -0.5, 'anger': -0.7, 'fear': -0.3
        }
        emotion_label = emotion_result.get('label', 'neutral').lower()
        emotion_score = emotion_scores.get(emotion_label, 0.0) * emotion_result.get('score', 0.5) * 0.2

        # PRIMARY: Use VADER as main source (most reliable for conversational text)
        # If VADER is very neutral (neu > 0.8), FORCE neutral regardless of compound score
        # This handles computer voices and factual text correctly
        if vader_neu > 0.7:
            # High neutral ratio - text is mostly neutral (computer voice, factual, etc.)
            # If neutral ratio is dominant, treat as neutral regardless of compound score
            if vader_neu > 0.85:
                # Very high neutral - force to neutral (0.0)
                overall_score = 0.0  # Completely neutral
            elif vader_neu > 0.8:
                # High neutral - heavily reduce compound score
                overall_score = vader_score * 0.1
            else:
                # Moderate neutral - reduce compound score significantly
                overall_score = vader_score * 0.2
        elif abs(vader_score) < 0.1:
            # VADER says neutral - trust it
            overall_score = vader_score * 0.95
        else:
            # VADER has clear sentiment - use it with small adjustments
            overall_score = (vader_score * 0.85 + bert_score * 0.1 + emotion_score * 0.05)

        # Clamp score to realistic range
        overall_score = max(-1.0, min(1.0, overall_score))

        # Categorize sentiment with realistic thresholds
        # For high-neutral text, use stricter thresholds to avoid false positives
        if vader_neu > 0.7:
            # High neutral text - use much stricter thresholds
            # This prevents computer voices from being classified as positive/negative
            if overall_score > 0.2:
                sentiment_label = "Positive"
            elif overall_score < -0.2:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
        else:
            # Normal thresholds
            if overall_score > 0.15:
                sentiment_label = "Positive"
            elif overall_score < -0.15:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

        return {
            'score': round(overall_score, 3),
            'label': sentiment_label,
            'confidence': min(abs(overall_score) * 2.5, 1.0)  # More realistic confidence
        }

    def _calculate_conversation_metrics(self, speaker_sentiments: Dict) -> Dict:
        """Calculate conversation-level sentiment metrics."""
        metrics = {
            'total_speakers': len(speaker_sentiments),
            'speaker_metrics': {},
            'conversation_trend': [],
            'sentiment_volatility': 0.0
        }

        all_scores = []

        for speaker, sentiments in speaker_sentiments.items():
            scores = [s['overall_sentiment']['score'] for s in sentiments]
            labels = [s['overall_sentiment']['label'] for s in sentiments]
            all_scores.extend(scores)

            # Calculate ratios based on LABELS (more reliable than scores)
            positive_count = sum(1 for label in labels if label == 'Positive')
            negative_count = sum(1 for label in labels if label == 'Negative')
            neutral_count = sum(1 for label in labels if label == 'Neutral')

            total_segments = len(sentiments)

            metrics['speaker_metrics'][speaker] = {
                'avg_sentiment': np.mean(scores) if scores else 0.0,
                'sentiment_range': max(scores) - min(scores) if len(scores) > 1 else 0.0,
                'segment_count': total_segments,
                'positive_ratio': positive_count / total_segments if total_segments > 0 else 0.0,
                'negative_ratio': negative_count / total_segments if total_segments > 0 else 0.0,
                'neutral_ratio': neutral_count / total_segments if total_segments > 0 else 0.0
            }

        if all_scores:
            metrics['conversation_trend'] = all_scores
            metrics['sentiment_volatility'] = np.std(all_scores) if len(all_scores) > 1 else 0.0
            metrics['overall_avg_sentiment'] = np.mean(all_scores)

        return metrics

    def _empty_sentiment_result(self) -> Dict:
        """Return empty sentiment result for error cases."""
        return {
            'text': '',
            'vader_scores': {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0},
            'bert_sentiment': {'label': 'NEUTRAL', 'score': 0.5},
            'emotion': {'label': 'neutral', 'score': 1.0},
            'cs_analysis': {'labels': [], 'scores': []},
            'nlp_features': {},
            'overall_sentiment': {'score': 0.0, 'label': 'Neutral', 'confidence': 0.0},
            'timestamp': None
        }

    def generate_sentiment_report(self, conversation_analysis: Dict) -> str:
        """
        Generate a human-readable sentiment report.

        Args:
            conversation_analysis: Conversation sentiment analysis results.

        Returns:
            Human-readable sentiment report string.
        """
        report = "## Sentiment Analysis Report\n\n"

        metrics = conversation_analysis.get('conversation_metrics', {})
        speaker_metrics = metrics.get('speaker_metrics', {})

        # Overall conversation sentiment
        overall_avg = metrics.get('overall_avg_sentiment', 0)
        volatility = metrics.get('sentiment_volatility', 0)

        report += f"**Overall Conversation Sentiment:** {overall_avg:.2f}\n"
        report += f"**Sentiment Volatility:** {volatility:.2f}\n\n"

        # Per-speaker analysis
        for speaker, speaker_data in speaker_metrics.items():
            report += f"### {speaker}\n"
            report += f"- Average Sentiment: {speaker_data['avg_sentiment']:.2f}\n"
            report += f"- Positive Ratio: {speaker_data['positive_ratio']:.1%}\n"
            report += f"- Negative Ratio: {speaker_data['negative_ratio']:.1%}\n"
            if 'neutral_ratio' in speaker_data:
                report += f"- Neutral Ratio: {speaker_data['neutral_ratio']:.1%}\n"
            report += f"- Sentiment Range: {speaker_data['sentiment_range']:.2f}\n\n"

        return report


__all__ = ["AdvancedSentimentAnalyzer"]
