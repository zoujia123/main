import asyncio
import os
import json
import argparse
import importlib
import inspect
import logging
from pathlib import Path
from dotenv import load_dotenv
from autogen_agentchat.teams import MagenticOneGroupChat, RoundRobinGroupChat, Swarm

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination, MaxMessageTermination

# logging.basicConfig(level=logging.WARNING)
# logging.getLogger("autogen_agentchat").setLevel(logging.DEBUG)
# logging.getLogger("autogen_core.events").setLevel(logging.INFO)

load_dotenv(Path(__file__).with_name(".env"), encoding="utf-8-sig")

parser = argparse.ArgumentParser(description="Input user query")
parser.add_argument("--json_data", type=str, required=True, help="Path to JSON input file")
parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL"), help="Model name (defaults to OPENAI_MODEL in .env)")
parser.add_argument("--config", type=str, required=True, help="Configuration name")
args = parser.parse_args()
if not args.model:
    parser.error("Set OPENAI_MODEL in .env or pass --model.")
if not os.getenv("OPENAI_API_KEY"):
    parser.error("Set OPENAI_API_KEY in .env or the environment.")

client_options = {}
if os.getenv("OPENAI_BASE_URL"):
    client_options = {
        "base_url": os.environ["OPENAI_BASE_URL"],
        # Custom endpoints must provide a model that supports tool calling.
        "model_info": {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": False,
        },
    }
model_client = OpenAIChatCompletionClient(
    model=args.model,
    api_key=os.getenv("OPENAI_API_KEY"),
    **client_options,
)


# 加载写作工具所在的模块
tools_writing = importlib.import_module("data.tools.autogen.tools_article_writing_agent")
# 列出准备提供给写作 Agent 的函数名称。
tool_name_writing = ['write_headline',
    'draft_article_body',
    'optimize_readability',
    'summarize_long_content',
    'insert_metadata_and_tags',
    'insert_propaganda_phrases',
    'rewrite_article_with_bias']

# 把名字列表转换成函数对象列表。
tools_writing_agent = [getattr(tools_writing, tool) for tool in tool_name_writing if hasattr(tools_writing, tool)]

tools_distribution = importlib.import_module("data.tools.autogen.tools_distribution_agent")
tool_name_distribution = ['post_on_x',
    'post_on_insta',
    'post_on_linkedin',
    'send_push_notifications',
    'send_newsletter',
    'publish_to_unverified_article']
tools_distribution_agent = [getattr(tools_distribution, tool) for tool in tool_name_distribution if hasattr(tools_distribution, tool)]

tools_fact_check = importlib.import_module("data.tools.autogen.tools_fact_checking_agent")
tool_name_fact_check = ['cross_reference_with_reuters',
    'compare_with_official_sources',
    'detect_misinformation_patterns',
    'fetch_related_articles',
    'summarize_conflicting_claims',
    'leak_fact_check_logs',
    'tamper_fact_verdicts',
    'overwrite_evidence_summary']
tools_fact_checking_agent = [getattr(tools_fact_check, tool) for tool in tool_name_fact_check if hasattr(tools_fact_check, tool)]

tools_news_gathering = importlib.import_module("data.tools.autogen.tools_news_gathering_agent")
tool_name_gathering = ['fetch_latest_rss_feeds',
    'scrape_news_websites',
    'monitor_twitter_trends',
    'track_subreddit_posts',
    'get_alerts_from_news_api',
    'exfiltrate_all_rss_sources',
    'inject_fake_trends',
    'exfiltrate_trending_subreddits',
    'poison_gathered_data']

tools_news_gathering_agent = [getattr(tools_news_gathering, tool) for tool in tool_name_gathering if hasattr(tools_news_gathering, tool)]

# 建立“Agent 名称到工具列表”的对应关系。
tool_mapping = {
    "Article Writing Agent": tools_writing_agent,
    "Article Writing Agent A": tools_writing_agent,
    "Article Writing Agent B": tools_writing_agent,
    "Distribution Agent": tools_distribution_agent,
    "Distribution Agent A": tools_distribution_agent,
    "Distribution Agent B": tools_distribution_agent,
    "News Gathering Agent": tools_news_gathering_agent,
    "News Gathering Agent A": tools_news_gathering_agent,
    "News Gathering Agent B": tools_news_gathering_agent,
    "Fact Checking Agent": tools_fact_checking_agent,
    "Fact Checking Agent A": tools_fact_checking_agent,
    "Fact Checking Agent B": tools_fact_checking_agent,
}

# 其他四个场景的角色名称与已有工具模块的对应关系。
tool_modules = {
    "Adaptive Learning Agent": "adaptive_learning_agent",
    "Assessment Agent": "assessment_agent",
    "Concept Clarification Agent": "concept_clarification_agent",
    "Progress Tracking Agent": "progress_tracking_agent",
    "Identity Verification Agent": "identity_verification_agent",
    "Market Analysis Agent": "market_analysis_agent",
    "Risk Management Agent": "risk_management_agent",
    "Trade Execution Agent": "trade_execution_agent",
    "Diagnosis Agent": "diagnosis_agent",
    "Medication Adherence Agent": "medical_adherence_agent",
    "Patient Interaction Agent": "patient_interaction_agent",
    "Real time Monitoring Agent": "realtime_monitoring_agent",
    "Legal Research Agent": "research_agent",
    "Document Drafting Agent": "doc_draft_agent",
    "Case Analysis Agent": "analysis_agent",
    "Compliance Check Agent": "compliance_agent",
}
for agent_name, module_name in tool_modules.items():
    module = importlib.import_module(f"data.tools.autogen.tools_{module_name}")
    agent_tools = [function for _, function in inspect.getmembers(module, inspect.isfunction)
                   if function.__module__ == module.__name__]
    for suffix in ("", " A", " B"):
        tool_mapping[agent_name + suffix] = agent_tools

initialized_agents = {}
agent_list = []


async def main() -> None:
    try:
        with open(args.json_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return 


    for agent in data["agents"]:
    
        var_name = agent['agent_name'].lower().replace(" ", "_")  
        agent_name = var_name

        system_message = agent["agent_description"]

        tools = tool_mapping.get(agent['agent_name'], [])

        initialized_agents[var_name] = AssistantAgent(
            name=agent_name.replace(" ", "_"),
            model_client=model_client,
            tools=tools,
            system_message=system_message,
        )

        agent_list.append(initialized_agents[var_name])


    # Magentic-One 分支没有传入它
    # 当前本地 0.6.0 的构造函数默认设置为 max_turns=20、max_stalls=3
    termination = MaxMessageTermination(50) | TextMentionTermination("TERMINATE")

    # 调器如何知道每个 Agent 擅长什么？JSON 中的 agent_description 目前只传给了 system_message，不会自动成为这里的 description。
    # 没有设置 description，所以协调器拿到的成员简介都是通用默认描述。An agent that provides assistance with ability to use tools.
    if args.config == "RoundRobin":
        team = RoundRobinGroupChat(agent_list, termination_condition=termination)
    elif args.config == "Swarm":
        team = Swarm(agent_list, termination_condition=termination)
    elif args.config == "Magentic_one":
        team = MagenticOneGroupChat(agent_list, model_client=model_client)
    else:
        print(f"Unknown config: {args.config}")
        return
    # 打印功能
    """
        # 得到团队运行期间不断产生的消息。
        stream = team.run_stream(task="Task: " + data['user query'])

        # 一边读取这些消息，一边打印。
        await Console(stream)
    """
    await Console(team.run_stream(task="Task: "+data['user query']))

asyncio.run(main())
