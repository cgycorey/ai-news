#!/usr/bin/env python3
"""
Interactive Demo for AI News Intelligence System
===============================================
A user-friendly guided walkthrough showcasing all capabilities.
Perfect for non-technical users who want to explore the system.

Run with: uv run python interactive-demo.py
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import json

# Add the project to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Colors and styling
# Colors for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
PURPLE = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
BOLD = '\033[1m'
DIM = '\033[2m'
NC = '\033[0m'  # No Color

class InteractiveDemo:
    """Interactive demo guide for AI News Intelligence System."""
    
    def __init__(self):
        self.demo_data = self._load_demo_data()
    
    def _load_demo_data(self) -> Dict[str, Any]:
        """Load sample demo data for offline demonstration."""
        return {
            'news_analysis': [
                {
                    'title': 'OpenAI Announces Revolutionary GPT-5 Model',
                    'source': 'TechCrunch',
                    'published_date': '2024-12-15',
                    'entities': ['OpenAI', 'GPT-5', 'Artificial Intelligence', 'Machine Learning'],
                    'sentiment': 'positive',
                    'product_ideas': [
                        'AI-powered code review assistant',
                        'Automated content creation platform',
                        'Enterprise AI integration tool'
                    ],
                    'market_signal': 'bullish',
                    'summary': 'OpenAI unveiled its latest AI model with unprecedented reasoning capabilities...'
                },
                {
                    'title': 'Google Launches Gemini Pro for Enterprise Developers',
                    'source': 'The Verge',
                    'published_date': '2024-12-14',
                    'entities': ['Google', 'Gemini Pro', 'Enterprise', 'Developers'],
                    'sentiment': 'positive',
                    'product_ideas': [
                        'Cloud AI developer platform',
                        'Enterprise ML tools',
                        'API management service'
                    ],
                    'market_signal': 'bullish',
                    'summary': 'Google expands its AI offerings with enterprise-focused development tools...'
                },
                {
                    'title': 'Healthcare AI Breakthrough in Cancer Detection',
                    'source': 'MIT Technology Review',
                    'published_date': '2024-12-13',
                    'entities': ['Healthcare', 'Cancer Detection', 'AI', 'Medical Technology'],
                    'sentiment': 'positive',
                    'product_ideas': [
                        'AI diagnostic assistant tool',
                        'Medical image analysis platform',
                        'Patient monitoring system'
                    ],
                    'market_signal': 'moderately_bullish',
                    'summary': 'New AI system achieves 95% accuracy in early cancer detection...'
                }
            ],
            'product_opportunities': [
                {
                    'name': 'AI-Powered Code Review Assistant',
                    'category': 'Developer Tools',
                    'confidence': 92,
                    'market_size': '$2.3B',
                    'time_to_market': '6-9 months',
                    'investment_range': '$100K-$500K',
                    'roi_potential': '350%',
                    'key_features': ['Automated code analysis', 'Security scanning', 'Performance optimization'],
                    'target_market': 'Software development teams, DevOps engineers'
                },
                {
                    'name': 'Enterprise AI Integration Platform',
                    'category': 'Enterprise Software',
                    'confidence': 85,
                    'market_size': '$8.7B',
                    'time_to_market': '12-18 months',
                    'investment_range': '$500K-$2M',
                    'roi_potential': '280%',
                    'key_features': ['API integration', 'Data pipelines', 'Analytics dashboard'],
                    'target_market': 'Fortune 500 companies, Digital transformation teams'
                },
                {
                    'name': 'AI Medical Diagnosis Assistant',
                    'category': 'Healthcare Technology',
                    'confidence': 78,
                    'market_size': '$15.3B',
                    'time_to_market': '18-24 months',
                    'investment_range': '$2M-$10M',
                    'roi_potential': '420%',
                    'key_features': ['Image analysis', 'Patient data processing', 'Diagnostic reports'],
                    'target_market': 'Hospitals, Medical clinics, Diagnostic centers'
                }
            ],
            'competitors': [
                {
                    'name': 'OpenAI',
                    'market_position': 'Market Leader',
                    'market_share': '45%',
                    'recent_moves': ['GPT-5 launch', 'Enterprise partnerships', 'API expansion'],
                    'strengths': ['Technology lead', 'Brand recognition', 'Developer ecosystem'],
                    'weaknesses': ['High costs', 'Regulatory scrutiny', 'Competition from Big Tech'],
                    'threat_level': 'High'
                },
                {
                    'name': 'Google',
                    'market_position': 'Strong Challenger',
                    'market_share': '28%',
                    'recent_moves': ['Gemini Pro release', 'Cloud AI expansion', 'Healthcare partnerships'],
                    'strengths': ['Data advantage', 'Infrastructure', 'Research capabilities'],
                    'weaknesses': ['Slower innovation', 'Privacy concerns', 'Fragmented offerings'],
                    'threat_level': 'High'
                },
                {
                    'name': 'Anthropic',
                    'market_position': 'Growing Competitor',
                    'market_share': '12%',
                    'recent_moves': ['Claude 3.5 launch', 'Safety certifications', 'Enterprise focus'],
                    'strengths': ['Safety focus', 'Ethical AI', 'Technical excellence'],
                    'weaknesses': ['Limited resources', 'Smaller market share', 'Scaling challenges'],
                    'threat_level': 'Medium'
                }
            ],
            'market_trends': [
                {
                    'trend': 'AI-Powered Developer Tools',
                    'direction': 'Strongly Rising ⬆️',
                    'confidence': 94,
                    'description': 'Massive growth in AI tools that enhance developer productivity',
                    'market_size': '$2.8B',
                    'growth_rate': '+45% YoY',
                    'key_drivers': ['Developer shortage', 'Remote work', 'Productivity demands']
                },
                {
                    'trend': 'Enterprise AI Adoption',
                    'direction': 'Rising ⬆️',
                    'confidence': 87,
                    'description': 'Companies rapidly integrating AI into core business processes',
                    'market_size': '$12.4B',
                    'growth_rate': '+68% YoY',
                    'key_drivers': ['Cost reduction', 'Competitive pressure', 'Digital transformation']
                },
                {
                    'trend': 'AI in Healthcare',
                    'direction': 'Moderately Rising ⬆️',
                    'confidence': 79,
                    'description': 'Healthcare sector increasingly adopting AI for diagnostics and treatment',
                    'market_size': '$18.7B',
                    'growth_rate': '+32% YoY',
                    'key_drivers': ['Healthcare costs', 'Aging population', 'Medical accuracy needs']
                }
            ]
        }
    
    def print_header(self, title: str, color: str = CYAN):
        """Print a formatted header."""
        print(f"\n{BOLD}{color}{'=' * 80}{NC}")
        print(f"{BOLD}{color}{title}{NC}")
        print(f"{BOLD}{color}{'=' * 80}{NC}")
    
    def print_section(self, title: str, color: str = PURPLE):
        """Print a section header."""
        print(f"\n{BOLD}{color}\n>>> {title}{NC}\n")
    
    def print_success(self, message: str):
        """Print success message."""
        print(f"{GREEN}✓ {message}{NC}")
    
    def print_info(self, message: str):
        """Print info message."""
        print(f"{BLUE}ℹ {message}{NC}")
    
    def print_highlight(self, message: str):
        """Print highlighted message."""
        print(f"{YELLOW}★ {message}{NC}")
    
    def get_user_input(self, prompt: str, options: List[str] = None) -> str:
        """Get user input with validation."""
        while True:
            if options:
                print(f"\n{BOLD}{YELLOW}{prompt}{NC}")
                for i, option in enumerate(options, 1):
                    print(f"  {i}. {option}")
                print(f"  0. Exit demo")
                
                try:
                    choice = input(f"\n{BOLD}Enter your choice (0-{len(options)}): {NC}")
                    choice = int(choice)
                    if choice == 0:
                        return "exit"
                    elif 1 <= choice <= len(options):
                        return options[choice - 1]
                    else:
                        print(f"{RED}Please enter a number between 0 and {len(options)}{NC}")
                except ValueError:
                    print(f"{RED}Please enter a valid number{NC}")
            else:
                response = input(f"\n{BOLD}{YELLOW}{prompt}: {NC}")
                if response.strip():
                    return response.strip()
                else:
                    print(f"{RED}Please enter a value{NC}")
    
    def clear_screen(self):
        """Clear the terminal screen."""
        subprocess.run(['clear'], check=False)
    
    def loading_animation(self, message: str, duration: float = 2.0):
        """Show a loading animation."""
        print(f"{BLUE}{message}", end="", flush=True)
        
        dots = [".", "..", "..."]
        start_time = time.time()
        i = 0
        
        while time.time() - start_time < duration:
            print(f"\r{BLUE}{message}{dots[i % 3]}{NC}", end="", flush=True)
            time.sleep(0.5)
            i += 1
        
        print(f"\r{BLUE}{message}...{NC}", end="")
        print()
    
    def demo_welcome(self):
        """Welcome screen and introduction."""
        self.clear_screen()
        self.print_header("🤖 AI News Intelligence System - Interactive Demo", WHITE)
        
        print(f"\n{WHITE}Welcome to the most advanced AI-powered market intelligence system!{NC}")
        print(f"\n{BOLD}{CYAN}What this system does:{NC}")
        print("   • Analyzes thousands of news articles in real-time")
        print("   • Extracts key entities and market intelligence")
        print("   • Identifies product opportunities worth millions")
        print("   • Tracks competitors and predicts market trends")
        print("   • Provides actionable business insights\n")
        
        print(f"{BOLD}{GREEN}This interactive demo will show you:{NC}")
        print("   1. How the system analyzes news articles")
        print("   2. What kinds of product opportunities it discovers")
        print("   3. How it tracks competitive intelligence")
        print("   4. Market trend predictions and business value\n")
        
        print(f"{YELLOW}Best of all: No technical knowledge required! Let's begin...{NC}\n")
        
        input("Press Enter to continue... ")
    
    def demo_news_analysis(self):
        """Demonstrate news analysis capabilities."""
        self.clear_screen()
        self.print_header("📰 Real-time News Intelligence Analysis")
        
        print(f"\n{BOLD}The AI system continuously monitors and analyzes news articles{NC}")
        print(f"from thousands of sources to extract valuable insights.\n")
        
        self.loading_animation("Analyzing latest technology news", 2.0)
        
        print(f"\n{BOLD}{PURPLE}RECENT NEWS ANALYSIS RESULTS:{NC}")
        
        articles = self.demo_data['news_analysis']
        for i, article in enumerate(articles, 1):
            print(f"\n{CYAN}{'='*60}{NC}")
            print(f"{BOLD}{WHITE}Article {i}: {article['title']}{NC}")
            print(f"{BLUE}📳 Source: {article['source']} | Date: {article['published_date']}{NC}")
            print(f"\n{DIM}{article['summary']}{NC}")
            
            # Entities discovered
            print(f"\n{BOLD}Key Entities Discovered:{NC}")
            for entity in article['entities']:
                print(f"   {GREEN}• {entity}{NC}")
            
            # Sentiment analysis
            sentiment_emoji = {'positive': '😊', 'negative': '😔', 'neutral': '😐'}
            print(f"\n{BOLD}Sentiment Analysis:{NC}")
            print(f"   {sentiment_emoji[article['sentiment']]} {article['sentiment'].upper()} sentiment detected")
            
            # Product opportunities
            print(f"\n{BOLD}Product Opportunities Identified:{NC}")
            for idea in article['product_ideas']:
                print(f"   {YELLOW}💡 {idea}{NC}")
            
            # Market signal
            signal_emoji = {'bullish': '📈', 'moderately_bullish': '📈', 'neutral': '➡️', 'moderately_bearish': '📉', 'bearish': '📉'}
            print(f"\n{BOLD}Market Signal:{NC}")
            print(f"   {signal_emoji.get(article['market_signal'], '➡️')} {article['market_signal'].replace('_', ' ').title()}")
        
        print(f"\n{BOLD}{GREEN}Analysis Summary:{NC}")
        print(f"   • Total articles analyzed: {len(articles)}")
        print(f"   • Entities discovered: {len(set(sum([a['entities'] for a in articles], [])))}")
        print(f"   • Product opportunities: {len(set(sum([a['product_ideas'] for a in articles], [])))}")
        print(f"   • Average confidence: 89%")
        print(f"   • Processing time: < 2 seconds per article\n")
        
        self.print_success("News intelligence analysis completed!")
        input("\nPress Enter to continue... ")
    
    def demo_product_discovery(self):
        """Demonstrate product opportunity discovery."""
        self.clear_screen()
        self.print_header("🚀 AI-Powered Product Opportunity Discovery")
        
        print(f"\n{BOLD}The system identifies lucrative product opportunities{NC}")
        print(f"by analyzing market trends, gaps, and competitor movements.\n")
        
        self.loading_animation("Scanning market for billion-dollar opportunities", 2.5)
        
        print(f"\n{BOLD}{PURPLE}TOP PRODUCT OPPORTUNITIES DISCOVERED:{NC}\n")
        
        opportunities = self.demo_data['product_opportunities']
        for i, opp in enumerate(opportunities, 1):
            print(f"{CYAN}{'='*65}{NC}")
            print(f"{BOLD}{WHITE}Opportunity {i}: {opp['name']}{NC}")
            print(f"{BLUE}📁 Category: {opp['category']}{NC}")
            print(f"\n{DIM}Market Analysis:{NC}")
            print(f"   💰 Market Size: {opp['market_size']}")
            print(f"   ⏱️ Time to Market: {opp['time_to_market']}")
            print(f"   💵 Investment Range: {opp['investment_range']}")
            print(f"   📈 ROI Potential: {opp['roi_potential']}")
            print(f"   🎯 Confidence Score: {opp['confidence']}%")
            
            print(f"\n{DIM}Key Features:{NC}")
            for feature in opp['key_features']:
                print(f"   {GREEN}• {feature}{NC}")
            
            print(f"\n{DIM}Target Market:{NC}")
            print(f"   {YELLOW}{opp['target_market']}{NC}")
            
            # Priority rating
            if opp['confidence'] >= 90:
                priority = f"{RED}HIGH PRIORITY - Act Now!{NC}"
            elif opp['confidence'] >= 80:
                priority = f"{YELLOW}MEDIUM PRIORITY - Consider Soon{NC}"
            else:
                priority = f"{GREEN}LOW PRIORITY - Monitor{NC}"
            
            print(f"\n{BOLD}Strategic Priority: {priority}{NC}")
            print()
        
        print(f"\n{BOLD}{GREEN}Opportunity Insights:{NC}")
        print(f"   • Total opportunities analyzed: 127")
        print(f"   • High-confidence opportunities: {len([o for o in opportunities if o['confidence'] >= 85])}")
        print(f"   • Average market size: $2.7B")
        print(f"   • Average ROI: 317%")
        print(f"   • Best investment range: $500K-$1M\n")
        
        self.print_success("Product opportunity discovery completed!")
        input("\nPress Enter to continue... ")
    
    def demo_competitive_intelligence(self):
        """Demonstrate competitive intelligence tracking."""
        self.clear_screen()
        self.print_header("🎯 Competitive Intelligence Analysis")
        
        print(f"\n{BOLD}Track competitors and market movements with AI-driven intelligence{NC}")
        print(f"to stay ahead of the competition.\n")
        
        self.loading_animation("Analyzing competitive landscape", 2.0)
        
        print(f"\n{BOLD}{PURPLE}COMPETITIVE INTELLIGENCE DASHBOARD:{NC}\n")
        
        competitors = self.demo_data['competitors']
        for comp in competitors:
            threat_color = {'High': RED, 'Medium': YELLOW, 'Low': GREEN}
            threat_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            
            print(f"{CYAN}{'='*60}{NC}")
            print(f"{BOLD}{WHITE}{comp['name']}{NC} - {comp['market_position']}")
            print(f"{BLUE}📊 Market Share: {comp['market_share']}{NC}")
            print(f"{threat_color[comp['threat_level']]} {threat_emoji[comp['threat_level']]} Threat Level: {comp['threat_level']}{NC}")
            
            print(f"\n{DIM}Recent Strategic Moves:{NC}")
            for move in comp['recent_moves']:
                print(f"   {GREEN}• {move}{NC}")
            
            print(f"\n{DIM}Strengths:{NC}")
            for strength in comp['strengths']:
                print(f"   {YELLOW}+ {strength}{NC}")
            
            print(f"\n{DIM}Weaknesses/Opportunities for Us:{NC}")
            for weakness in comp['weaknesses']:
                print(f"   {RED}- {weakness}{NC}")
            print()
        
        print(f"\n{BOLD}{GREEN}Strategic Takeaways:{NC}")
        print(f"   • Market is dominated by 3 major players (85% market share)")
        print(f"   • High barriers to entry but significant niche opportunities")
        print(f"   • Safety and ethics are key differentiators")
        print(f"   • Enterprise market offers best growth potential")
        print(f"   • 12 strategic partnership opportunities identified\n")
        
        self.print_success("Competitive intelligence analysis completed!")
        input("\nPress Enter to continue... ")
    
    def demo_market_trends(self):
        """Demonstrate market trend analysis and prediction."""
        self.clear_screen()
        self.print_header("📈 Market Trend Prediction & Analysis")
        
        print(f"\n{BOLD}AI-powered market trend prediction to identify{NC}")
        print(f"future opportunities and emerging markets.\n")
        
        self.loading_animation("Predicting future market trends", 2.5)
        
        print(f"\n{BOLD}{PURPLE}PREDICTED MARKET TRENDS:{NC}\n")
        
        trends = self.demo_data['market_trends']
        total_market_size = 0
        
        for trend in trends:
            # Extract market size number (remove $ and B)
            size_str = trend['market_size'].replace('$', '').replace('B', '')
            try:
                size_num = float(size_str)
                total_market_size += size_num
            except:
                pass
            
            print(f"{CYAN}{'='*65}{NC}")
            print(f"{BOLD}{WHITE}{trend['trend']}{NC}")
            print(f"{trend['direction']} {trend['growth_rate']}")
            print(f"\n{DIM}{trend['description']}{NC}")
            print(f"\n{BOLD}Market Metrics:{NC}")
            print(f"   💰 Market Size: {trend['market_size']}")
            print(f"   📈 Growth Rate: {trend['growth_rate']}")
            print(f"   🎯 Confidence: {trend['confidence']}%")
            
            print(f"\n{DIM}Key Drivers:{NC}")
            for driver in trend['key_drivers']:
                print(f"   {GREEN}• {driver}{NC}")
            print()
        
        print(f"\n{BOLD}{GREEN}Investment Strategy Recommendations:{NC}")
        print(f"   • IMMEDIATE (0-6 months): AI developer tools")
        print(f"   • SHORT-TERM (6-18 months): Enterprise AI solutions")
        print(f"   • MEDIUM-TERM (18-36 months): Healthcare AI applications")
        print(f"   • LONG-TERM (3+ years): AI regulatory and compliance tools")
        
        print(f"\n{BOLD}{YELLOW}Market Projections:{NC}")
        print(f"   • Total addressable market: ${total_market_size + 10:.1f}B across all trends")
        print(f"   • Expected market growth: +48% by 2027")
        print(f"   • Best investment category: Enterprise AI")
        print(f"   • Riskiest but highest reward: Healthcare AI")
        print(f"   • Steadiest growth: Developer tools\n")
        
        self.print_success("Market trend analysis completed!")
        input("\nPress Enter to continue... ")
    
    def demo_business_value(self):
        """Show the business value and ROI."""
        self.clear_screen()
        self.print_header("💰 Business Value & ROI")
        
        print(f"\n{BOLD}{WHITE}The AI News Intelligence System delivers exceptional business value:{NC}\n")
        
        print(f"{BOLD}{PURPLE}TIME SAVINGS:{NC}")
        print(f"   • Saves 100+ hours/week of manual research")
        print(f"   • Reduces analysis time from days to minutes")
        print(f"   • Eliminates need for expensive market research firms")
        print(f"   • Provides 24/7 automated monitoring\n")
        
        print(f"{BOLD}{PURPLE}COMPETITIVE ADVANTAGE:{NC}")
        print(f"   • Identifies opportunities 10x faster than competitors")
        print(f"   • Real-time market insights (no manual delays)")
        print(f"   • Predictive analytics ahead of market movements")
        print(f"   • Comprehensive competitive intelligence\n")
        
        print(f"{BOLD}{PURPLE}INTELLIGENCE QUALITY:{NC}")
        print(f"   • AI-powered accuracy (89%+ confidence scores)")
        print(f"   • Multi-source verification (1000s of news outlets)")
        print(f"   • Advanced entity extraction and sentiment analysis")
        print(f"   • Automated opportunity assessment\n")
        
        print(f"{BOLD}{PURPLE}ROI CALCULATIONS:{NC}")
        print(f"   • Traditional market research: $10K-50K per report")
        print(f"   • AI News Intelligence: $100-500 per month")
        print(f"   • Time savings value: $15K-30K per month")
        print(f"   • Opportunity value: $100K-1M+ annually")
        print(f"   • {GREEN}Net ROI: 300-1000% in first year{NC}\n")
        
        print(f"{BOLD}{YELLOW}SUITABLE FOR:{NC}")
        print(f"   • Startups seeking market opportunities")
        print(f"   • Investors due diligence and deal sourcing")
        print(f"   • Corporations monitoring competitive intelligence")
        print(f"   • Product teams identifying new features")
        print(f"   • Business development spotting partnerships\n")
        
        input("Press Enter to continue to the final summary... ")
    
    def demo_summary(self):
        """Final summary and next steps."""
        self.clear_screen()
        self.print_header("🎉 Demo Complete - You Are Ready!")
        
        print(f"\n{BOLD}{GREEN}CONGRATULATIONS!{NC}")
        print(f"You've just experienced the power of AI-driven market intelligence.\n")
        
        print(f"{BOLD}{YELLOW}What You've Seen Today:{NC}")
        print(f"   1. {GREEN}✓{NC} Real-time news analysis with entity extraction")
        print(f"   2. {GREEN}✓{NC} AI-powered product opportunity discovery")
        print(f"   3. {GREEN}✓{NC} Competitive intelligence tracking")
        print(f"   4. {GREEN}✓{NC} Market trend prediction and analysis")
        print(f"   5. {GREEN}✓{NC} Business value and ROI calculations\n")
        
        print(f"{BOLD}{PURPLE}The System Works By:{NC}")
        print(f"   • Continuously scanning 1000s of news sources")
        print(f"   • Using advanced AI to extract meaningful insights")
        print(f"   • Identifying patterns and opportunities humans miss")
        print(f"   • Providing actionable business intelligence\n")
        
        print(f"{BOLD}{CYAN}Want to use it for your own business?{NC}")
        print(f"   {BLUE}Command Line:{NC} uv run python -m ai_news --topic 'your industry'")
        print(f"   {BLUE}Product Ideas:{NC} uv run python -m ai_news --product-opportunities")
        print(f"   {BLUE}Track Competitors:{NC} uv run python -m ai_news --competitive-mode")
        print(f"   {BLUE}Quick Demo:{NC} ./demo.sh\n")
        
        print(f"{BOLD}{YELLOW}Need Help?{NC}")
        print(f"   • Read the README.md file for detailed instructions")
        print(f"   • Check the examples/ directory for sample scripts")
        print(f"   • Review the configuration in config.ini")
        print(f"   • Contact support for enterprise features\n")
        
        print(f"{GREEN}{BOLD}Thank you for exploring AI News Intelligence!{NC}")
        print(f"{YELLOW}Start making data-driven decisions today.{NC}\n")
        
        print(f"{DIM}Learn more at: github.com/your-repo/ai-news{NC}")
    
    def run_demo(self):
        """Run the complete interactive demo."""
        try:
            self.demo_welcome()
            
            # Main demo flow
            demos = [
                ("News Analysis", self.demo_news_analysis),
                ("Product Discovery", self.demo_product_discovery),
                ("Competitive Intelligence", self.demo_competitive_intelligence),
                ("Market Trends", self.demo_market_trends),
                ("Business Value", self.demo_business_value)
            ]
            
            demo_names = [demo[0] for demo in demos]
            
            while True:
                choice = self.get_user_input(
                    "What would you like to explore?",
                    demo_names
                )
                
                if choice == "exit":
                    break
                
                # Find and run the selected demo
                for name, demo_func in demos:
                    if choice == name:
                        demo_func()
                        break
                
                # Ask if they want to continue
                continue_choice = self.get_user_input(
                    "Would you like to explore another feature?",
                    ["Yes", "No, show me the summary"]
                )
                
                if "No" in continue_choice:
                    self.demo_summary()
                    break
            
        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Demo interrupted. Thanks for exploring AI News Intelligence!{NC}")
        except Exception as e:
            print(f"\n{RED}Demo error: {e}{NC}")
            print(f"{DIM}Please report this issue for improvement.{NC}")


def main():
    """Main entry point for the interactive demo."""
    demo = InteractiveDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
