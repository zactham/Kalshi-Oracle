#!/usr/bin/env python3
"""
Comprehensive test runner for Kalshi Oracle Bot
Runs all test scripts and provides a summary report
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_test(test_name, test_file):
    """Run a single test and return results"""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"📁 File: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} - PASSED ({duration:.2f}s)")
            return True, result.stdout, result.stderr
        else:
            print(f"❌ {test_name} - FAILED ({duration:.2f}s)")
            print(f"Error: {result.stderr}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} - TIMEOUT (60s)")
        return False, "", "Test timed out after 60 seconds"
    except Exception as e:
        print(f"💥 {test_name} - ERROR: {e}")
        return False, "", str(e)

def main():
    """Run all tests and provide summary"""
    print("🚀 Kalshi Oracle Bot - Comprehensive Test Suite")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version}")
    
    # Define all tests
    tests = [
        ("Kalshi API Integration", "test_kalshi_api.py"),
        ("Dynamic Proposals", "test_dynamic_proposals.py"),
        ("Headline Isolation", "test_headline_isolation.py"),
        ("Error Handling", "test_error_handling.py"),
        ("News Diversity", "test_news_diversity.py"),
    ]
    
    # Track results
    results = []
    passed = 0
    failed = 0
    
    # Run each test
    for test_name, test_file in tests:
        if os.path.exists(test_file):
            success, stdout, stderr = run_test(test_name, test_file)
            results.append({
                'name': test_name,
                'file': test_file,
                'success': success,
                'stdout': stdout,
                'stderr': stderr
            })
            
            if success:
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  {test_name} - SKIPPED (file not found: {test_file})")
            results.append({
                'name': test_name,
                'file': test_file,
                'success': False,
                'stdout': "",
                'stderr': f"File not found: {test_file}"
            })
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    # Detailed results
    if failed > 0:
        print(f"\n❌ FAILED TESTS:")
        for result in results:
            if not result['success']:
                print(f"  • {result['name']}: {result['stderr'][:100]}...")
    
    print(f"\n🏁 Test suite completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
