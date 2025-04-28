from cctv_monitor import CCTVMonitor
import argparse
import sys
import signal
import time
import os


def handle_signal(signum, frame):
    """Handle termination signals gracefully."""
    print("\n[SIGNAL] Received termination signal. Shutting down gracefully...")
    sys.exit(0)


def main():
    # Set up signal handlers for graceful termination
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(description='CCTV Monitoring System')
    parser.add_argument('--mode', choices=['process', 'report', 'query'], required=True,
                        help='Operation mode: process (process image file), report (generate report), or query (search past events)')
    parser.add_argument('--query', help='Query string for searching past events')
    parser.add_argument('--storage', default=None,
                        help='Custom storage path for data (default: ~/CCTV_Monitoring)')
    parser.add_argument('--file', help='Path to image file to process')

    args = parser.parse_args()

    try:
        print("[STARTUP] Initializing CCTV Monitor...")
        monitor = CCTVMonitor(storage_path=args.storage)

        if args.mode == 'process':
            if not args.file:
                print("[ERROR] Please provide an image file path using --file")
                sys.exit(1)
                
            if not os.path.exists(args.file):
                print(f"[ERROR] File does not exist: {args.file}")
                sys.exit(1)
                
            print(f"[PROCESS] Processing image file: {args.file}")
            try:
                caption, analysis = monitor.process_uploaded_image(args.file)
                print("\n[RESULT] Image Processing Results:")
                print("-" * 50)
                print(f"Caption: {caption}")
                print(f"Suspicious: {analysis.get('is_suspicious', False)}")
                if analysis.get('is_suspicious', False):
                    print(f"Reason: {analysis.get('reason', 'Unknown')}")
                    print(f"Confidence: {analysis.get('confidence', 0.0)}")
                print("-" * 50)
                print("[PROCESS] Image processed and data saved.")
            except Exception as e:
                print(f"\n[ERROR] Processing error: {str(e)}")
                import traceback
                traceback.print_exc()

        elif args.mode == 'report':
            print("[REPORT] Generating daily report...")
            report = monitor.generate_report()
            print("\n[REPORT] Report generated successfully!")
            print("\n[REPORT] Report Preview:")
            print("-" * 50)
            print(report[:500] + "..." if len(report) > 500 else report)
            print("-" * 50)
            print(f"\n[REPORT] Full report saved to: {monitor.data_dir}/daily_report.txt")

        elif args.mode == 'query':
            if not args.query:
                print("[ERROR] Please provide a query string using --query")
                sys.exit(1)

            print(f"[QUERY] Searching for: {args.query}")
            result = monitor.query_events(args.query)
            print("\n[QUERY] Search Results:")
            print("-" * 50)
            print(result)
            print("-" * 50)

    except ValueError as e:
        print(f"[ERROR] Configuration error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()