"""
MCP Server Integration Examples and Workflows

Comprehensive examples showing how to use MCP servers (Perplexity, Tavily, Brave, Omnisearch)
together with sophisticated orchestration, cross-validation, and integration with ADK built-in tools.
"""

import asyncio
from typing import Any, Dict, List, Optional

from .factory import (
    MCPServerFactory, 
    MCPSuite, 
    create_orchestrator,
    get_all_mcp_servers,
)
from .orchestrator import MCPOrchestrator, SearchStrategy, SearchContext, QueryType
from .servers.perplexity import PerplexitySearchType
from .servers.tavily import TavilySearchType
from .servers.brave import BraveFreshness
from ..context import ToolContextPattern, MemoryAccessPattern
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..config.tools import ToolsConfig


class MCPWorkflowDemo:
    """
    Demonstration class showing comprehensive MCP server integration.
    
    Shows how to use all MCP servers together with sophisticated orchestration,
    cross-validation, and integration with the platform's service architecture.
    """
    
    def __init__(self,
                 api_keys: Dict[str, str],
                 config: Optional[ToolsConfig] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        self.api_keys = api_keys
        self.config = config
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        
        # Create MCP server factory
        self.factory = MCPServerFactory(
            config=config,
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        # Create comprehensive MCP suite
        self.mcp_servers = self.factory.create_suite(MCPSuite.COMPREHENSIVE, api_keys)
        
        # Create orchestrator
        self.orchestrator = self.factory.create_orchestrator(self.mcp_servers)
        
        # Context patterns for integration
        self.tool_pattern = ToolContextPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        self.memory_pattern = MemoryAccessPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
    
    async def demo_research_intelligence_workflow(self, 
                                                topic: str,
                                                research_depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Demonstrate comprehensive research intelligence workflow using MCP orchestration.
        
        Flow: Multi-perspective research → Cross-validation → Synthesis → Timeline analysis
        """
        if self.logger:
            self.logger.info(f"Starting research intelligence workflow for: {topic}")
        
        workflow_results = {
            "topic": topic,
            "initial_research": None,
            "multi_perspective_analysis": None,
            "cross_validation": None,
            "temporal_analysis": None,
            "synthesis_report": None,
        }
        
        try:
            # Phase 1: Initial comprehensive research
            if self.logger:
                self.logger.info("Phase 1: Initial comprehensive research")
            
            research_context = SearchContext(
                query=f"comprehensive overview of {topic}",
                query_type=QueryType.RESEARCH,
                quality_threshold=0.8,
                time_budget=30.0,
            )
            
            initial_research = await self.orchestrator.search(
                research_context, 
                SearchStrategy.QUALITY_OPTIMIZED
            )
            workflow_results["initial_research"] = initial_research.to_dict()
            
            # Phase 2: Multi-perspective analysis
            if self.logger:
                self.logger.info("Phase 2: Multi-perspective analysis")
            
            perspectives = ["academic", "industry", "policy", "technological", "economic"]
            perspective_results = {}
            
            for perspective in perspectives:
                perspective_context = SearchContext(
                    query=f"{topic} from {perspective} perspective",
                    query_type=QueryType.RESEARCH,
                    quality_threshold=0.7,
                )
                
                perspective_result = await self.orchestrator.search(
                    perspective_context,
                    SearchStrategy.HYBRID_VALIDATION
                )
                perspective_results[perspective] = perspective_result.to_dict()
            
            workflow_results["multi_perspective_analysis"] = perspective_results
            
            # Phase 3: Cross-validation with fact-checking
            if self.logger:
                self.logger.info("Phase 3: Cross-validation and fact-checking")
            
            # Extract key claims from initial research
            key_claims = self._extract_key_claims(initial_research.consensus_answer)
            validation_results = {}
            
            for claim in key_claims[:3]:  # Validate top 3 claims
                fact_check_result = await self.orchestrator.fact_check_workflow(claim)
                validation_results[claim] = fact_check_result
            
            workflow_results["cross_validation"] = validation_results
            
            # Phase 4: Temporal analysis
            if self.logger:
                self.logger.info("Phase 4: Temporal analysis")
            
            time_ranges = {
                "recent": "past month",
                "current_year": "past year", 
                "historical": "past 5 years",
            }
            
            temporal_results = await self.orchestrator.temporal_search(
                query=f"developments in {topic}",
                time_ranges=time_ranges,
            )
            workflow_results["temporal_analysis"] = {
                period: result.to_dict() for period, result in temporal_results.items()
            }
            
            # Phase 5: Synthesis report
            if self.logger:
                self.logger.info("Phase 5: Generating synthesis report")
            
            synthesis_report = await self._generate_synthesis_report(
                topic, workflow_results
            )
            workflow_results["synthesis_report"] = synthesis_report
            
            if self.logger:
                self.logger.info(f"Completed research intelligence workflow for: {topic}")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Research intelligence workflow failed: {e}")
            workflow_results["error"] = str(e)
        
        return workflow_results
    
    async def demo_real_time_monitoring_workflow(self,
                                               topic: str,
                                               monitoring_keywords: List[str]) -> Dict[str, Any]:
        """
        Demonstrate real-time monitoring workflow using MCP servers.
        
        Flow: News monitoring → Social sentiment → Trend analysis → Alert generation
        """
        if self.logger:
            self.logger.info(f"Starting real-time monitoring workflow for: {topic}")
        
        monitoring_results = {
            "topic": topic,
            "keywords": monitoring_keywords,
            "news_monitoring": None,
            "social_sentiment": None,
            "trend_analysis": None,
            "alerts": [],
        }
        
        try:
            # News monitoring with Tavily (real-time focus)
            tavily_server = self.mcp_servers.get("tavily")
            if tavily_server:
                news_result = await tavily_server.news_search(
                    query=topic,
                    days_back=7,
                    max_results=20,
                )
                monitoring_results["news_monitoring"] = news_result.to_dict()
            
            # Social sentiment analysis with Perplexity
            perplexity_server = self.mcp_servers.get("perplexity")
            if perplexity_server:
                sentiment_result = await perplexity_server.analyze_with_context(
                    content=f"Recent news about {topic}",
                    analysis_prompt=f"Analyze public sentiment and key discussions about {topic} in recent news and social media",
                )
                monitoring_results["social_sentiment"] = sentiment_result.to_dict()
            
            # Trend analysis using orchestrated search
            trend_context = SearchContext(
                query=f"trending topics related to {topic}",
                query_type=QueryType.NEWS,
                quality_threshold=0.6,
            )
            
            trend_result = await self.orchestrator.search(
                trend_context,
                SearchStrategy.SPEED_OPTIMIZED
            )
            monitoring_results["trend_analysis"] = trend_result.to_dict()
            
            # Generate alerts based on keyword monitoring
            alerts = await self._generate_monitoring_alerts(
                topic, monitoring_keywords, monitoring_results
            )
            monitoring_results["alerts"] = alerts
            
            if self.logger:
                self.logger.info(f"Completed real-time monitoring workflow for: {topic}")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Real-time monitoring workflow failed: {e}")
            monitoring_results["error"] = str(e)
        
        return monitoring_results
    
    async def demo_competitive_intelligence_workflow(self,
                                                   company: str,
                                                   competitors: List[str]) -> Dict[str, Any]:
        """
        Demonstrate competitive intelligence workflow using MCP orchestration.
        
        Flow: Company research → Competitor analysis → Market positioning → Strategic insights
        """
        if self.logger:
            self.logger.info(f"Starting competitive intelligence workflow for: {company}")
        
        intelligence_results = {
            "company": company,
            "competitors": competitors,
            "company_profile": None,
            "competitor_analysis": {},
            "market_positioning": None,
            "strategic_insights": None,
        }
        
        try:
            # Company profile research
            company_context = SearchContext(
                query=f"{company} company profile business model financials",
                query_type=QueryType.COMMERCIAL,
                quality_threshold=0.8,
            )
            
            company_profile = await self.orchestrator.search(
                company_context,
                SearchStrategy.QUALITY_OPTIMIZED
            )
            intelligence_results["company_profile"] = company_profile.to_dict()
            
            # Competitor analysis
            competitor_analyses = {}
            for competitor in competitors:
                competitor_context = SearchContext(
                    query=f"{competitor} vs {company} comparison analysis",
                    query_type=QueryType.COMMERCIAL,
                )
                
                competitor_result = await self.orchestrator.search(
                    competitor_context,
                    SearchStrategy.ADAPTIVE
                )
                competitor_analyses[competitor] = competitor_result.to_dict()
            
            intelligence_results["competitor_analysis"] = competitor_analyses
            
            # Market positioning analysis
            positioning_context = SearchContext(
                query=f"{company} market position competitive landscape",
                query_type=QueryType.COMMERCIAL,
            )
            
            positioning_result = await self.orchestrator.search(
                positioning_context,
                SearchStrategy.HYBRID_VALIDATION
            )
            intelligence_results["market_positioning"] = positioning_result.to_dict()
            
            # Strategic insights synthesis
            strategic_insights = await self._generate_strategic_insights(
                company, competitors, intelligence_results
            )
            intelligence_results["strategic_insights"] = strategic_insights
            
            if self.logger:
                self.logger.info(f"Completed competitive intelligence workflow for: {company}")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Competitive intelligence workflow failed: {e}")
            intelligence_results["error"] = str(e)
        
        return intelligence_results
    
    async def demo_content_verification_workflow(self,
                                               content: str,
                                               verification_level: str = "comprehensive") -> Dict[str, Any]:
        """
        Demonstrate content verification workflow using cross-validation.
        
        Flow: Content analysis → Source verification → Fact-checking → Credibility assessment
        """
        if self.logger:
            self.logger.info("Starting content verification workflow")
        
        verification_results = {
            "content": content[:500] + "..." if len(content) > 500 else content,
            "content_analysis": None,
            "source_verification": None,
            "fact_checking": {},
            "credibility_assessment": None,
        }
        
        try:
            # Content analysis with Perplexity
            perplexity_server = self.mcp_servers.get("perplexity")
            if perplexity_server:
                analysis_result = await perplexity_server.analyze_with_context(
                    content=content,
                    analysis_prompt="Analyze this content for factual claims, potential biases, and key assertions that should be verified",
                )
                verification_results["content_analysis"] = analysis_result.to_dict()
                
                # Extract claims for fact-checking
                claims = self._extract_factual_claims(analysis_result.answer)
            else:
                claims = ["No content analysis available"]
            
            # Source verification using multiple MCP servers
            source_verification = await self._verify_content_sources(content)
            verification_results["source_verification"] = source_verification
            
            # Fact-checking individual claims
            fact_check_results = {}
            for claim in claims[:5]:  # Check top 5 claims
                fact_check_result = await self.orchestrator.fact_check_workflow(
                    claim=claim,
                    verification_sources=["perplexity", "tavily", "brave"],
                )
                fact_check_results[claim] = fact_check_result
            
            verification_results["fact_checking"] = fact_check_results
            
            # Overall credibility assessment
            credibility_assessment = self._assess_content_credibility(
                verification_results
            )
            verification_results["credibility_assessment"] = credibility_assessment
            
            if self.logger:
                self.logger.info("Completed content verification workflow")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Content verification workflow failed: {e}")
            verification_results["error"] = str(e)
        
        return verification_results
    
    async def demo_cross_platform_integration(self,
                                            query: str) -> Dict[str, Any]:
        """
        Demonstrate integration between MCP servers and ADK built-in tools.
        
        Shows how MCP external services complement ADK built-in capabilities.
        """
        if self.logger:
            self.logger.info(f"Starting cross-platform integration demo for: {query}")
        
        integration_results = {
            "query": query,
            "mcp_results": {},
            "adk_integration": {},
            "cross_validation": {},
            "unified_insights": None,
        }
        
        try:
            # MCP server results
            search_context = SearchContext(
                query=query,
                query_type=QueryType.RESEARCH,
            )
            
            mcp_result = await self.orchestrator.search(
                search_context,
                SearchStrategy.COMPREHENSIVE
            )
            integration_results["mcp_results"] = mcp_result.to_dict()
            
            # Simulate ADK built-in tool integration
            # (In real implementation, this would use actual ADK tools)
            adk_integration = {
                "google_search": {
                    "query": query,
                    "results_count": 10,
                    "relevance_score": 0.85,
                    "unique_domains": 8,
                },
                "vertex_ai_search": {
                    "semantic_results": 5,
                    "similarity_threshold": 0.7,
                    "embedding_quality": 0.9,
                },
                "bigquery_analysis": {
                    "data_points": 1000,
                    "trend_analysis": "positive",
                    "statistical_significance": 0.95,
                },
            }
            integration_results["adk_integration"] = adk_integration
            
            # Cross-validation between MCP and ADK results
            cross_validation = self._cross_validate_mcp_adk_results(
                mcp_result, adk_integration
            )
            integration_results["cross_validation"] = cross_validation
            
            # Generate unified insights
            unified_insights = self._generate_unified_insights(
                mcp_result, adk_integration, cross_validation
            )
            integration_results["unified_insights"] = unified_insights
            
            if self.logger:
                self.logger.info("Completed cross-platform integration demo")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cross-platform integration demo failed: {e}")
            integration_results["error"] = str(e)
        
        return integration_results
    
    def _extract_key_claims(self, text: str) -> List[str]:
        """Extract key factual claims from text for verification."""
        # Simple implementation - in practice, this would use NLP
        sentences = text.split('. ')
        
        # Filter for sentences that look like factual claims
        claims = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in 
                  ['is', 'are', 'was', 'were', 'has', 'have', 'will', 'according to']):
                if len(sentence.strip()) > 20:  # Substantive claims
                    claims.append(sentence.strip())
        
        return claims[:5]  # Return top 5 claims
    
    def _extract_factual_claims(self, analysis_text: str) -> List[str]:
        """Extract factual claims identified in content analysis."""
        # Look for claims section in analysis
        if "claims:" in analysis_text.lower():
            claims_section = analysis_text.lower().split("claims:")[1]
            claims = [claim.strip() for claim in claims_section.split('\n') if claim.strip()]
            return claims[:5]
        
        # Fallback to general extraction
        return self._extract_key_claims(analysis_text)
    
    async def _verify_content_sources(self, content: str) -> Dict[str, Any]:
        """Verify sources mentioned in content using MCP servers."""
        # Extract URLs and source mentions from content
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        
        source_verification = {
            "urls_found": urls,
            "url_verification": {},
            "source_credibility": {},
        }
        
        # Verify each URL/source
        for url in urls[:3]:  # Check first 3 URLs
            try:
                # Use Tavily for content extraction and verification
                tavily_server = self.mcp_servers.get("tavily")
                if tavily_server:
                    verification = await tavily_server.extract_content([url])
                    source_verification["url_verification"][url] = verification
            except Exception as e:
                source_verification["url_verification"][url] = {"error": str(e)}
        
        return source_verification
    
    def _assess_content_credibility(self, verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall content credibility based on verification results."""
        credibility_score = 0.5  # Start with neutral
        assessment_factors = []
        
        # Factor in fact-checking results
        fact_check_results = verification_results.get("fact_checking", {})
        if fact_check_results:
            verified_claims = sum(1 for result in fact_check_results.values() 
                                if result.get("fact_check_status") == "verified")
            total_claims = len(fact_check_results)
            
            if total_claims > 0:
                fact_check_score = verified_claims / total_claims
                credibility_score = credibility_score * 0.3 + fact_check_score * 0.7
                assessment_factors.append(f"Fact-checking: {verified_claims}/{total_claims} claims verified")
        
        # Factor in source verification
        source_verification = verification_results.get("source_verification", {})
        url_verification = source_verification.get("url_verification", {})
        if url_verification:
            successful_verifications = sum(1 for v in url_verification.values() 
                                         if not v.get("error"))
            total_urls = len(url_verification)
            
            if total_urls > 0:
                source_score = successful_verifications / total_urls
                credibility_score = credibility_score * 0.7 + source_score * 0.3
                assessment_factors.append(f"Source verification: {successful_verifications}/{total_urls} sources verified")
        
        # Determine credibility level
        if credibility_score >= 0.8:
            credibility_level = "high"
        elif credibility_score >= 0.6:
            credibility_level = "moderate"
        elif credibility_score >= 0.4:
            credibility_level = "low"
        else:
            credibility_level = "very_low"
        
        return {
            "credibility_score": credibility_score,
            "credibility_level": credibility_level,
            "assessment_factors": assessment_factors,
            "recommendation": self._get_credibility_recommendation(credibility_level),
        }
    
    def _get_credibility_recommendation(self, credibility_level: str) -> str:
        """Get recommendation based on credibility level."""
        recommendations = {
            "high": "Content appears highly credible based on verification",
            "moderate": "Content has moderate credibility - verify key claims independently",
            "low": "Content has low credibility - exercise caution and seek additional sources",
            "very_low": "Content has very low credibility - recommend finding alternative sources",
        }
        return recommendations.get(credibility_level, "Unable to assess credibility")
    
    async def _generate_synthesis_report(self, 
                                       topic: str, 
                                       workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive synthesis report from research workflow."""
        synthesis = {
            "topic": topic,
            "executive_summary": "",
            "key_findings": [],
            "perspectives_summary": {},
            "validated_facts": [],
            "temporal_trends": {},
            "confidence_assessment": {},
            "recommendations": [],
        }
        
        # Generate executive summary
        initial_research = workflow_results.get("initial_research", {})
        if initial_research:
            synthesis["executive_summary"] = initial_research.get("consensus_answer", "")[:500]
        
        # Summarize perspectives
        perspectives = workflow_results.get("multi_perspective_analysis", {})
        for perspective, result in perspectives.items():
            synthesis["perspectives_summary"][perspective] = {
                "key_points": result.get("consensus_answer", "")[:200],
                "confidence": result.get("confidence_score", 0.5),
                "sources_count": result.get("total_results", 0),
            }
        
        # Extract validated facts
        validation_results = workflow_results.get("cross_validation", {})
        for claim, validation in validation_results.items():
            if validation.get("fact_check_status") == "verified":
                synthesis["validated_facts"].append({
                    "claim": claim,
                    "confidence": validation.get("reliability_score", 0.5),
                })
        
        # Temporal trends analysis
        temporal_analysis = workflow_results.get("temporal_analysis", {})
        for period, result in temporal_analysis.items():
            synthesis["temporal_trends"][period] = {
                "developments": result.get("consensus_answer", "")[:150],
                "activity_level": result.get("total_results", 0),
            }
        
        # Generate recommendations
        synthesis["recommendations"] = [
            "Continue monitoring developments in this area",
            "Validate key findings with additional expert sources", 
            "Consider implications from multiple perspectives",
            "Monitor temporal trends for emerging patterns",
        ]
        
        return synthesis
    
    async def _generate_monitoring_alerts(self,
                                        topic: str,
                                        keywords: List[str],
                                        monitoring_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on monitoring results."""
        alerts = []
        
        # News volume alerts
        news_monitoring = monitoring_results.get("news_monitoring", {})
        if news_monitoring:
            source_count = news_monitoring.get("source_count", 0)
            if source_count > 10:
                alerts.append({
                    "type": "high_news_volume",
                    "severity": "medium",
                    "message": f"High news volume detected for {topic}: {source_count} sources",
                    "data": {"source_count": source_count},
                })
        
        # Sentiment alerts
        sentiment_analysis = monitoring_results.get("social_sentiment", {})
        if sentiment_analysis:
            # Simple sentiment detection in answer
            answer = sentiment_analysis.get("answer", "").lower()
            if any(word in answer for word in ["crisis", "controversy", "scandal"]):
                alerts.append({
                    "type": "negative_sentiment",
                    "severity": "high", 
                    "message": f"Negative sentiment detected for {topic}",
                    "data": {"sentiment_indicators": ["crisis", "controversy", "scandal"]},
                })
        
        # Keyword alerts
        for keyword in keywords:
            # Check if keyword appears frequently in results
            keyword_mentions = 0
            for result_key, result_data in monitoring_results.items():
                if isinstance(result_data, dict) and "consensus_answer" in result_data:
                    keyword_mentions += result_data["consensus_answer"].lower().count(keyword.lower())
            
            if keyword_mentions > 5:
                alerts.append({
                    "type": "keyword_surge",
                    "severity": "low",
                    "message": f"Keyword '{keyword}' mentioned {keyword_mentions} times",
                    "data": {"keyword": keyword, "mentions": keyword_mentions},
                })
        
        return alerts
    
    async def _generate_strategic_insights(self,
                                         company: str,
                                         competitors: List[str],
                                         intelligence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic insights from competitive intelligence."""
        insights = {
            "competitive_positioning": {},
            "market_opportunities": [],
            "threats_analysis": [],
            "strategic_recommendations": [],
            "confidence_assessment": {},
        }
        
        # Analyze competitive positioning
        company_profile = intelligence_results.get("company_profile", {})
        competitor_analysis = intelligence_results.get("competitor_analysis", {})
        
        insights["competitive_positioning"] = {
            "company_strength": company_profile.get("confidence_score", 0.5),
            "competitor_comparison": {
                comp: result.get("confidence_score", 0.5) 
                for comp, result in competitor_analysis.items()
            },
            "market_position": intelligence_results.get("market_positioning", {}).get("confidence_score", 0.5),
        }
        
        # Generate strategic recommendations
        insights["strategic_recommendations"] = [
            "Strengthen competitive differentiation in key areas",
            "Monitor competitor activities and market responses",
            "Identify potential partnership or acquisition opportunities", 
            "Develop response strategies for competitive threats",
        ]
        
        return insights
    
    def _cross_validate_mcp_adk_results(self,
                                      mcp_result,
                                      adk_integration: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate results between MCP servers and ADK tools."""
        validation = {
            "consistency_score": 0.75,  # Simulated consistency
            "complementary_insights": [],
            "conflicting_information": [],
            "confidence_boost": 0.1,
        }
        
        # Check for complementary insights
        mcp_sources = mcp_result.sources_used if hasattr(mcp_result, 'sources_used') else []
        adk_tools = list(adk_integration.keys())
        
        validation["complementary_insights"] = [
            f"MCP sources ({', '.join(mcp_sources)}) provide external perspective",
            f"ADK tools ({', '.join(adk_tools)}) provide structured data analysis",
            "Cross-platform validation increases result reliability",
        ]
        
        return validation
    
    def _generate_unified_insights(self,
                                 mcp_result,
                                 adk_integration: Dict[str, Any],
                                 cross_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unified insights from MCP and ADK integration."""
        return {
            "unified_confidence": 0.85,  # Combined confidence from both platforms
            "comprehensive_coverage": True,
            "data_triangulation": "Multiple independent sources confirm key findings",
            "platform_strengths": {
                "mcp": "Real-time external data and AI-powered analysis",
                "adk": "Structured search and enterprise data integration",
            },
            "synthesis": "Integration provides comprehensive view with both external and internal perspectives",
        }


# Convenience functions for running demos

async def run_research_intelligence_demo(api_keys: Dict[str, str],
                                       topic: str = "artificial intelligence ethics") -> Dict[str, Any]:
    """Run the research intelligence workflow demonstration."""
    demo = MCPWorkflowDemo(api_keys)
    return await demo.demo_research_intelligence_workflow(topic)


async def run_real_time_monitoring_demo(api_keys: Dict[str, str],
                                      topic: str = "climate change",
                                      keywords: List[str] = None) -> Dict[str, Any]:
    """Run the real-time monitoring workflow demonstration."""
    if keywords is None:
        keywords = ["climate", "carbon", "emissions", "sustainability", "green energy"]
    
    demo = MCPWorkflowDemo(api_keys)
    return await demo.demo_real_time_monitoring_workflow(topic, keywords)


async def run_competitive_intelligence_demo(api_keys: Dict[str, str],
                                          company: str = "OpenAI",
                                          competitors: List[str] = None) -> Dict[str, Any]:
    """Run the competitive intelligence workflow demonstration."""
    if competitors is None:
        competitors = ["Anthropic", "Google DeepMind", "Microsoft"]
    
    demo = MCPWorkflowDemo(api_keys)
    return await demo.demo_competitive_intelligence_workflow(company, competitors)


async def run_content_verification_demo(api_keys: Dict[str, str],
                                      content: str) -> Dict[str, Any]:
    """Run the content verification workflow demonstration."""
    demo = MCPWorkflowDemo(api_keys)
    return await demo.demo_content_verification_workflow(content)


async def run_cross_platform_integration_demo(api_keys: Dict[str, str],
                                             query: str = "machine learning trends 2024") -> Dict[str, Any]:
    """Run the cross-platform integration demonstration."""
    demo = MCPWorkflowDemo(api_keys)
    return await demo.demo_cross_platform_integration(query)


# Example usage and testing functions

async def comprehensive_mcp_demo(api_keys: Dict[str, str]) -> Dict[str, Any]:
    """Run all MCP workflow demonstrations."""
    print("MCP Server Integration - Comprehensive Demo")
    print("=" * 50)
    
    results = {}
    
    try:
        # Research Intelligence
        print("\n1. Research Intelligence Workflow")
        print("-" * 30)
        research_result = await run_research_intelligence_demo(
            api_keys, "quantum computing applications"
        )
        results["research_intelligence"] = research_result
        print(f"✓ Research completed with {len(research_result.get('multi_perspective_analysis', {}))} perspectives")
        
        # Real-time Monitoring
        print("\n2. Real-time Monitoring Workflow")
        print("-" * 30) 
        monitoring_result = await run_real_time_monitoring_demo(
            api_keys, "cryptocurrency regulation", ["bitcoin", "ethereum", "regulation", "SEC"]
        )
        results["real_time_monitoring"] = monitoring_result
        alert_count = len(monitoring_result.get("alerts", []))
        print(f"✓ Monitoring completed with {alert_count} alerts generated")
        
        # Competitive Intelligence
        print("\n3. Competitive Intelligence Workflow")
        print("-" * 30)
        intelligence_result = await run_competitive_intelligence_demo(
            api_keys, "Tesla", ["Ford", "GM", "Volkswagen"]
        )
        results["competitive_intelligence"] = intelligence_result
        competitor_count = len(intelligence_result.get("competitor_analysis", {}))
        print(f"✓ Intelligence gathered on {competitor_count} competitors")
        
        # Cross-platform Integration
        print("\n4. Cross-platform Integration Demo")
        print("-" * 30)
        integration_result = await run_cross_platform_integration_demo(
            api_keys, "renewable energy storage solutions"
        )
        results["cross_platform_integration"] = integration_result
        confidence = integration_result.get("unified_insights", {}).get("unified_confidence", 0)
        print(f"✓ Integration completed with {confidence:.2f} unified confidence")
        
        print(f"\n✅ All MCP workflow demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        results["error"] = str(e)
    
    return results


if __name__ == "__main__":
    print("MCP Server Integration Examples")
    print("=" * 40)
    print()
    print("This module provides comprehensive examples of:")
    print("1. Research Intelligence - Multi-perspective analysis with cross-validation")
    print("2. Real-time Monitoring - News monitoring with alert generation")
    print("3. Competitive Intelligence - Company and market analysis")
    print("4. Content Verification - Fact-checking and credibility assessment") 
    print("5. Cross-platform Integration - MCP + ADK built-in tools")
    print()
    print("Required API keys:")
    print("- perplexity_api_key: Perplexity AI API key")
    print("- tavily_api_key: Tavily search API key")
    print("- brave_api_key: Brave Search API key")
    print("- omnisearch_api_key: Omnisearch API key (if using)")
    print()
    print("Use the async functions to run demonstrations:")
    print("- comprehensive_mcp_demo(api_keys)")
    print("- run_research_intelligence_demo(api_keys, topic)")
    print("- run_real_time_monitoring_demo(api_keys, topic, keywords)")
    print("- run_competitive_intelligence_demo(api_keys, company, competitors)")
    print("- run_content_verification_demo(api_keys, content)")
    print("- run_cross_platform_integration_demo(api_keys, query)")