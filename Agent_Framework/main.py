"""
Main entry point for the Gemini Agent Framework
"""

import asyncio
import argparse
from agent.live_agent import LiveAgent
from config.settings import DEFAULT_MODE, DEFAULT_WAKEWORD_MODEL

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Gemini Agent Framework")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="Video mode to use: none, camera, or screen",
        choices=["none", "camera", "screen"],
    )
    parser.add_argument(
        "--wakeword",
        type=str,
        default=DEFAULT_WAKEWORD_MODEL,
        help="Wake word model to use (path to model file)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (overrides default)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Gemini Assistant",
        help="Name for your assistant",
    )
    args = parser.parse_args()

    # Print welcome message
    print("\nStarting Gemini Agent Framework...")

    # Create a configuration dictionary to pass to the LiveAgent
    config = {
        "name": args.name,
        "video_mode": args.mode,
        "wakeword_model": args.wakeword
    }
    
    # Add API key if provided
    if args.api_key:
        config["api_key"] = args.api_key

    # Create and run the agent
    agent = LiveAgent(**config)
    
    async def run_agent():
        await agent.initialize()
        await agent.run()
    
    asyncio.run(run_agent())

if __name__ == "__main__":
    main()