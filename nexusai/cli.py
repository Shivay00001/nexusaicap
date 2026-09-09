"""NexusAI CLI — Command Line Interface."""
from __future__ import annotations
import sys, argparse
from nexusai.core.config import NexusConfig
from nexusai.agents.orchestrator import AgentOrchestrator

def main():
    parser = argparse.ArgumentParser(description="NexusAI - Autonomous Agent Ecosystem")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a task autonomously")
    run_parser.add_argument("task", help="The task to run")
    run_parser.add_argument("--provider", "-p", help="LLM Provider (e.g., openai, anthropic)")
    run_parser.add_argument("--model", "-m", help="Model name")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--provider", "-p", help="LLM Provider")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start Web/MCP Server")
    server_parser.add_argument("--port", type=int, default=8080, help="Web port")

    args = parser.parse_args()

    config = NexusConfig.from_env()
    if getattr(args, "provider", None):
        config.provider = args.provider
    if getattr(args, "model", None):
        config.model = args.model

    if args.command == "run":
        orchestrator = AgentOrchestrator(config=config, verbose=True)
        orchestrator.run(args.task)
    elif args.command == "chat":
        orchestrator = AgentOrchestrator(config=config, verbose=False)
        print(f"Starting chat with {config.provider}... (Type 'exit' to quit)")
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue
                print("\nNexusAI: Thinking...")
                response = orchestrator.chat(user_input)
                print(f"\nNexusAI: {response}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
    elif args.command == "server":
        print(f"Starting NexusAI Web Dashboard on port {args.port}...")
        from nexusai.web.app import run_server
        run_server(port=args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
