#!/usr/bin/env python3
"""
Simple script to check failed feature statistics
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from eeg2fx.feature_saver import get_failed_features_stats

def main():
    """Print failed feature statistics"""
    stats = get_failed_features_stats()
    
    print("=== Failed Feature Statistics ===")
    print(f"Total failed features: {stats['total_failed']}")
    
    if stats['failed_by_fxdef']:
        print("\nFailed features by fxdef_id:")
        for fxdef_id, count in list(stats['failed_by_fxdef'].items())[:10]:
            print(f"  fxdef_id {fxdef_id}: {count} failures")
    
    if stats['failed_by_recording']:
        print("\nFailed features by recording_id:")
        for recording_id, count in list(stats['failed_by_recording'].items())[:10]:
            print(f"  recording_id {recording_id}: {count} failures")
    
    if stats['common_errors']:
        print("\nCommon error messages:")
        for error, count in list(stats['common_errors'].items())[:5]:
            print(f"  {error}: {count} occurrences")
    
    if stats['total_failed'] == 0:
        print("\n✓ No failed features found!")

if __name__ == "__main__":
    main() 