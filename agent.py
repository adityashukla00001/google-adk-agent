from google.adk.agents import LlmAgent, Agent, LoopAgent
from google.adk.tools import google_search
from google.adk.tools import agent_tool

# MODEL="gemini-2.5-flash"
# MODEL="gemma-3-1b-it"
MODEL="gemini-2.0-flash-lite"
ROUNDS_OF_ARGUMENT = 3

from google.adk.agents import LlmAgent, Agent, LoopAgent
from google.adk.tools import google_search, agent_tool

# === Config ===
MODEL = "gemini-2.0-flash-lite"
SEARCH_MODEL = "gemini-2.5-flash"
ROUNDS = 3

# === Search Agent ===
search_agent = Agent(
    model=SEARCH_MODEL,
    name="search_agent",
    instruction="Use Google Search to find facts, data, and expert insights.",
    tools=[google_search]
)

# === Debater: Pro ===
pro_agent = LlmAgent(
    model=MODEL,
    name="pro_argument_agent",
    description="Presents arguments supporting the debate topic.",
    instruction="""
        You are ADVOCATE PRIME.

        Your mission: argue in **support** of the topic with precision, aggression, and clarity.
        • Present 2-3 strong points
        • Support each with logic and credible data (use the search tool)
        • Dismantle opponent's narrative
        • End with impactful statements

        Avoid emotional appeal. Be sharp, confident, and fact-based.
        """,
    tools=[agent_tool.AgentTool(agent=search_agent)],
)

# === Debater: Anti ===
anti_agent = LlmAgent(
    model=MODEL,
    name="anti_argument_agent",
    description="Presents arguments opposing the debate topic.",
    instruction="""
        You are OPPOSITION FORCE.

        Your mission: argue **against** the topic with precision, aggression, and clarity.
        • Refute opponent’s key points
        • Present counter-evidence and alternative perspectives
        • Use the search tool to ground your rebuttals in facts
        • Finish with strong, dismissive conclusions

        Discredit the topic logically. Be calm, ruthless, and effective.
        """,
    tools=[agent_tool.AgentTool(agent=search_agent)],
)

# === Judge ===
judge_agent = LlmAgent(
    model=MODEL,
    name="debate_judge",
    description="Evaluates both sides and gives verdict per round.",
    instruction="""
        You are SUPREME ARBITRATOR.

        Your task: fairly and critically judge the arguments presented.
        For each round:
        • Compare logic, structure, and factual strength of both sides
        • Validate claims using the search tool
        • Identify:
        - 🏆 Winner of the round (PRO or ANTI)
        - 📈 Strongest point (quote it)
        - 📉 Weakest point (quote it)
        - 🔢 Score: PRO [X/10] vs ANTI [X/10]

        Be neutral. Focus on who argued better, not which side is morally right.
        """,
    tools=[agent_tool.AgentTool(agent=search_agent)],
)

# === Game Loop ===
debate_loop = LoopAgent(
    name="debate_rounds_loop",
    description="Handles multiple rounds of arguments and judging.",
    sub_agents=[pro_agent, anti_agent, judge_agent],
    max_iterations=ROUNDS,
)

# === Orchestrator ===
root_agent = LlmAgent(
    model=MODEL,
    name="debate_orchestrator",
    description="Manages debate setup, execution, and final result.",
    instruction=f"""
        You are the host of the AI Debate Colosseum.

        1. Ask the user: “What topic should we debate?”
        2. Announce the match structure: {ROUNDS} rounds.
        3. Introduce the participants:
        • 🟢 PRO: {pro_agent.name}
        • 🔴 ANTI: {anti_agent.name}
        • ⚖️ JUDGE: {judge_agent.name}
        4. Start the battle using: {debate_loop.name}
        5. After all rounds, summarize:
        • Round-by-round verdicts
        • Total scores
        • Overall winner with reasons
        • Thank the participants and invite for another match

        Keep it dramatic, high-energy, but to the point.
        """,
    sub_agents=[debate_loop],
)
