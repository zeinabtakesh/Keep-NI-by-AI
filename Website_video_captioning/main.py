from cctv_monitor import CCTVMonitor
import argparse
import sys
import signal
import time

def handle_signal(signum, frame):
    """Handle termination signals gracefully."""
    print("\n[SIGNAL] Received termination signal. Shutting down gracefully...")
    sys.exit(0)

def main():
    # Set up signal handlers for graceful termination
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    parser = argparse.ArgumentParser(description='CCTV Monitoring System')
    parser.add_argument('--mode', choices=['monitor', 'report', 'query'], required=True,
                      help='Operation mode: monitor (live monitoring), report (generate report), or query (search past events)')
    parser.add_argument('--query', help='Query string for searching past events')
    parser.add_argument('--storage', default=None, 
                      help='Custom storage path for data (default: ~/CCTV_Monitoring)')
    
    args = parser.parse_args()
    
    try:
        print("[STARTUP] Initializing CCTV Monitor...")
        monitor = CCTVMonitor(storage_path=args.storage)
        
        if args.mode == 'monitor':
            print("[STARTUP] Starting CCTV monitoring... Press 'q' to quit.")
            try:
                monitor.run()
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] User interrupted monitoring.")
            except Exception as e:
                print(f"\n[ERROR] Monitoring error: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                print("[SHUTDOWN] Saving data and shutting down...")
                monitor.save_data()
                print("[SHUTDOWN] Monitoring stopped. Data saved.")
        
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