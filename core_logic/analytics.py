"""
Advanced Analytics for Customer Service Call Analysis (UI-agnostic).

This module provides conversation quality analysis and visualization capabilities.
All outputs are data-oriented (dicts, objects) - no UI formatting.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from config import Config

logger = logging.getLogger(__name__)


class ConversationAnalytics:
    """Advanced analytics for conversation analysis."""

    def __init__(self):
        """Initialize analytics engine."""
        self.setup_plotting()

    def setup_plotting(self) -> None:
        """Setup plotting configuration."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def analyze_conversation_quality(
        self,
        segments: List[Dict],
        sentiment_data: Dict
    ) -> Dict:
        """
        Comprehensive conversation quality analysis.

        Args:
            segments: List of conversation segments.
            sentiment_data: Sentiment analysis results.

        Returns:
            Dictionary with quality metrics.
        """
        # Calculate total speakers from segments
        unique_speakers = set(segment.get('speaker', 'Unknown') for segment in segments)
        total_speakers = len(unique_speakers)
        total_segments = len(segments)

        quality_metrics = {
            'conversation_flow': self._analyze_conversation_flow(segments),
            'response_patterns': self._analyze_response_patterns(segments),
            'sentiment_trends': self._analyze_sentiment_trends(sentiment_data),
            'total_speakers': total_speakers,
            'total_segments': total_segments,
            'quality_score': 0.0,
            'recommendations': []
        }

        # Calculate overall quality score
        quality_metrics['quality_score'] = self._calculate_quality_score(quality_metrics)

        # Generate recommendations
        quality_metrics['recommendations'] = self._generate_recommendations(quality_metrics)

        return quality_metrics

    def _analyze_conversation_flow(self, segments: List[Dict]) -> Dict:
        """
        Analyze conversation flow using only metrics that are reliably
        measurable from Whisper's sequential, non-overlapping segments.
        """
        if not segments:
            return {}

        unique_speakers = set(seg.get('speaker', 'Unknown') for seg in segments)

        # Count speaker changes
        speaker_changes = 0
        for i in range(1, len(segments)):
            if segments[i - 1].get('speaker') != segments[i].get('speaker'):
                speaker_changes += 1

        # Total conversation duration
        total_duration = 0.0
        if segments:
            total_duration = segments[-1].get('end', 0) - segments[0].get('start', 0)

        turns_per_min = (speaker_changes / (total_duration / 60.0)) if total_duration > 0 else 0.0

        return {
            'speaker_turns': speaker_changes,
            'turns_per_minute': round(turns_per_min, 1),
            'total_duration': round(total_duration, 1),
            'num_speakers': len(unique_speakers),
        }

    def _analyze_speaker_balance(self, segments: List[Dict]) -> Dict:
        """Analyze speaking time balance between speakers."""
        speaker_times = {}
        total_duration = 0

        for segment in segments:
            speaker = segment.get('speaker', 'Unknown')
            duration = segment.get('end', 0) - segment.get('start', 0)

            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            total_duration += duration

        if not speaker_times:
            return {}

        # Calculate balance metrics - more realistic for single speaker
        times = list(speaker_times.values())
        if len(times) == 1:
            # Single speaker gets poor balance score
            balance_score = 0.2
        else:
            # Multiple speakers - calculate actual balance
            balance_score = 1 - (max(times) - min(times)) / total_duration if total_duration > 0 else 0

        return {
            'speaker_times': speaker_times,
            'total_duration': total_duration,
            'balance_score': balance_score,
            'dominant_speaker': max(speaker_times, key=speaker_times.get),
            'speaking_ratios': {k: v / total_duration for k, v in speaker_times.items()}
        }

    def _analyze_response_patterns(self, segments: List[Dict]) -> Dict:
        """Analyze response patterns and turn-taking."""
        if not segments:
            return {}

        turn_lengths = []
        speaker_changes = 0

        for i, segment in enumerate(segments):
            duration = segment.get('end', 0) - segment.get('start', 0)
            turn_lengths.append(duration)

            if i > 0 and segment.get('speaker') != segments[i - 1].get('speaker'):
                speaker_changes += 1

        # More realistic efficiency calculation
        if len(segments) == 1:
            # Single segment = no turn-taking
            efficiency = 0.0
        else:
            # Multiple segments - calculate actual efficiency
            efficiency = speaker_changes / len(segments) if segments else 0

        return {
            'avg_turn_length': np.mean(turn_lengths),
            'turn_length_std': np.std(turn_lengths),
            'speaker_changes': speaker_changes,
            'turn_taking_efficiency': efficiency
        }

    def _analyze_sentiment_trends(self, sentiment_data: Dict) -> Dict:
        """Analyze sentiment trends over time."""
        if not sentiment_data or 'speaker_sentiments' not in sentiment_data:
            return {}

        trends = {}
        for speaker, sentiments in sentiment_data['speaker_sentiments'].items():
            scores = [s['overall_sentiment']['score'] for s in sentiments]
            timestamps = [s['timestamp'] for s in sentiments]

            # Calculate trend
            if len(scores) > 1:
                trend_slope = np.polyfit(timestamps, scores, 1)[0]
            else:
                trend_slope = 0

            trends[speaker] = {
                'sentiment_scores': scores,
                'timestamps': timestamps,
                'trend_slope': trend_slope,
                'avg_sentiment': np.mean(scores),
                'sentiment_volatility': np.std(scores)
            }

        return trends

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """
        Calculate overall conversation quality score (0-100).

        NOTE: This quality score is intended for multi-speaker customer service calls.
        If only one speaker is detected, the score isn't meaningful – return 0 with no penalties.
        """
        total_speakers = metrics.get('total_speakers', 1)
        if total_speakers < 2:
            return 0.0

        score = 30  # base score

        # Flow metrics (25 points) – based on pacing / turn-taking
        flow = metrics.get('conversation_flow', {})
        turns_per_min = flow.get('turns_per_minute', 0)
        # Good pacing: roughly 4-15 turns/min for a natural conversation
        if 4 <= turns_per_min <= 15:
            score += 25
        elif 2 <= turns_per_min < 4 or 15 < turns_per_min <= 25:
            score += 15
        else:
            score += 5

        # Response patterns (20 points)
        patterns = metrics.get('response_patterns', {})
        efficiency = patterns.get('turn_taking_efficiency', 0)
        score += min(20, efficiency * 25)

        # Sentiment trends (25 points)
        sentiment = metrics.get('sentiment_trends', {})
        if sentiment:
            avg_sentiments = [data['avg_sentiment'] for data in sentiment.values()]
            if avg_sentiments:
                avg_sentiment = np.mean(avg_sentiments)
                if avg_sentiment > 0.2:
                    score += 25  # Positive sentiment
                elif avg_sentiment < -0.2:
                    score += 5   # Negative sentiment
                else:
                    score += 15  # Neutral sentiment
        else:
            score += 5  # Default for no sentiment data

        # Penalty for very short conversations
        total_segments = metrics.get('total_segments', 1)
        if total_segments < 3:
            score = min(score, 50)

        return min(95, max(15, score))  # Realistic range 15-95

    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate improvement recommendations based on measurable metrics."""
        recommendations = []

        # If single speaker, call-quality recommendations don't apply.
        if metrics.get('total_speakers', 1) < 2:
            return recommendations

        # Pacing
        flow = metrics.get('conversation_flow', {})
        turns_per_min = flow.get('turns_per_minute', 0)
        if turns_per_min > 25:
            recommendations.append(
                "Very rapid speaker exchanges detected. Conversations may benefit from longer, more complete turns."
            )
        elif turns_per_min < 2 and metrics.get('total_segments', 0) > 3:
            recommendations.append(
                "Very few speaker turns detected. Encourage more interactive back-and-forth dialogue."
            )

        # Turn length
        patterns = metrics.get('response_patterns', {})
        if patterns.get('avg_turn_length', 0) > 30:
            recommendations.append(
                "Average speaking turns are long (>30 s). Shorter turns improve conversational engagement."
            )

        # Sentiment
        sentiment = metrics.get('sentiment_trends', {})
        for speaker, data in sentiment.items():
            if data.get('avg_sentiment', 0) < -0.3:
                recommendations.append(f"Negative sentiment detected for {speaker}. Consider follow-up.")

        return recommendations

    def create_visualizations(
        self,
        segments: List[Dict],
        sentiment_data: Dict,
        quality_metrics: Dict
    ) -> Dict[str, str]:
        """
        Create visualization plots.

        Args:
            segments: List of conversation segments.
            sentiment_data: Sentiment analysis results.
            quality_metrics: Quality metrics results.

        Returns:
            Dictionary mapping plot names to HTML strings.
        """
        plots = {}

        try:
            # Sentiment over time plot
            plots['sentiment_timeline'] = self._create_sentiment_timeline_plot(sentiment_data)

            # Speaker balance pie chart
            plots['speaker_balance'] = self._create_speaker_balance_plot(quality_metrics)

            # Conversation flow heatmap
            plots['conversation_flow'] = self._create_conversation_flow_plot(segments)

            # Quality metrics dashboard
            plots['quality_dashboard'] = self._create_quality_dashboard(quality_metrics)

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

        return plots

    def _create_sentiment_timeline_plot(self, sentiment_data: Dict) -> str:
        """Create sentiment timeline visualization."""
        if not sentiment_data or 'speaker_sentiments' not in sentiment_data:
            return ""

        fig = go.Figure()

        for speaker, sentiments in sentiment_data['speaker_sentiments'].items():
            timestamps = [s['timestamp'] for s in sentiments]
            scores = [s['overall_sentiment']['score'] for s in sentiments]

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=scores,
                mode='lines+markers',
                name=speaker,
                line=dict(width=2)
            ))

        fig.update_layout(
            title="Sentiment Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Sentiment Score",
            hovermode='x unified'
        )

        return fig.to_html(include_plotlyjs='cdn')

    def _create_speaker_balance_plot(self, quality_metrics: Dict) -> str:
        """Create speaker balance pie chart."""
        balance = quality_metrics.get('speaker_balance', {})
        speaker_times = balance.get('speaker_times', {})

        if not speaker_times:
            return ""

        fig = go.Figure(data=[go.Pie(
            labels=list(speaker_times.keys()),
            values=list(speaker_times.values()),
            hole=0.3
        )])

        fig.update_layout(title="Speaking Time Distribution")
        return fig.to_html(include_plotlyjs='cdn')

    def _create_conversation_flow_plot(self, segments: List[Dict]) -> str:
        """Create conversation flow visualization."""
        if not segments:
            return ""

        # Create timeline visualization
        fig = go.Figure()

        colors = px.colors.qualitative.Set3
        speakers = list(set(segment.get('speaker', 'Unknown') for segment in segments))
        speaker_colors = {speaker: colors[i % len(colors)] for i, speaker in enumerate(speakers)}

        for segment in segments:
            speaker = segment.get('speaker', 'Unknown')
            start = segment.get('start', 0)
            end = segment.get('end', 0)

            fig.add_trace(go.Scatter(
                x=[start, end, end, start, start],
                y=[speaker, speaker, speaker, speaker, speaker],
                fill='toself',
                fillcolor=speaker_colors[speaker],
                line=dict(color=speaker_colors[speaker]),
                name=speaker,
                showlegend=False
            ))

        fig.update_layout(
            title="Conversation Flow Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Speaker",
            hovermode='closest'
        )

        return fig.to_html(include_plotlyjs='cdn')

    def _create_quality_dashboard(self, quality_metrics: Dict) -> str:
        """Create quality metrics dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality Score', 'Interruption Rate', 'Balance Score', 'Response Time'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )

        # Quality score gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=quality_metrics.get('quality_score', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Quality Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ]
            }
        ), row=1, col=1)

        return fig.to_html(include_plotlyjs='cdn')

    def export_analysis_report(
        self,
        segments: List[Dict],
        sentiment_data: Dict,
        quality_metrics: Dict,
        output_path: str
    ) -> str:
        """
        Export comprehensive analysis report.

        Args:
            segments: List of conversation segments.
            sentiment_data: Sentiment analysis results.
            quality_metrics: Quality metrics results.
            output_path: Path to save the report.

        Returns:
            Path to the saved report.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'conversation_summary': {
                'total_segments': len(segments),
                'duration': max(segment.get('end', 0) for segment in segments) if segments else 0,
                'speakers': list(set(segment.get('speaker', 'Unknown') for segment in segments))
            },
            'quality_metrics': quality_metrics,
            'sentiment_analysis': sentiment_data,
            'recommendations': quality_metrics.get('recommendations', [])
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        return output_path


__all__ = ["ConversationAnalytics"]
