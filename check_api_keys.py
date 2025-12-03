#!/usr/bin/env python3
"""
Quick script to check which API keys are working
Run this to diagnose your API key issues
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

load_dotenv()

# Load all API keys
keys = []
i = 1
while True:
    key = os.getenv(f"GOOGLE_API_KEY_{i}")
    if key:
        keys.append((i, key))
        i += 1
    else:
        break

if not keys:
    print("❌ No API keys found in .env file!")
    print("Make sure your .env file has GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.")
    exit(1)

print(f"Found {len(keys)} API keys in .env file\n")
print("="*60)
print("Testing each API key...")
print("="*60)

working_keys = []
exhausted_keys = []

for index, key in keys:
    print(f"\n🔑 Testing Key #{index} ({key[:15]}...)")
    
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Make a simple test call
        response = model.generate_content("Say 'OK' if you can read this.")
        
        if response and response.text:
            print(f"   ✅ Key #{index} is WORKING")
            print(f"   Response: {response.text[:50]}")
            working_keys.append(index)
        else:
            print(f"   ⚠️ Key #{index} returned empty response")
            
    except Exception as e:
        error_str = str(e)
        
        if "429" in error_str or "quota" in error_str.lower() or "resource exhausted" in error_str.lower():
            print(f"   ❌ Key #{index} is EXHAUSTED (quota exceeded)")
            exhausted_keys.append(index)
        elif "invalid" in error_str.lower() or "api_key" in error_str.lower():
            print(f"   ❌ Key #{index} is INVALID")
        else:
            print(f"   ❌ Key #{index} ERROR: {error_str[:100]}")
    
    time.sleep(1)  # Rate limiting between tests

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✅ Working keys: {len(working_keys)}/{len(keys)}")
if working_keys:
    print(f"   Keys: {', '.join([f'#{k}' for k in working_keys])}")

print(f"❌ Exhausted keys: {len(exhausted_keys)}/{len(keys)}")
if exhausted_keys:
    print(f"   Keys: {', '.join([f'#{k}' for k in exhausted_keys])}")

print("\n💡 Recommendations:")
if len(working_keys) == 0:
    print("   ⚠️ ALL KEYS ARE EXHAUSTED!")
    print("   1. Wait 1-24 hours for quota reset")
    print("   2. Get more API keys from: https://aistudio.google.com/app/apikey")
    print("   3. Upgrade to paid tier: https://ai.google.dev/pricing")
    print("   4. Check usage at: https://ai.dev/usage?tab=rate-limit")
elif len(working_keys) < len(keys) / 2:
    print("   ⚠️ Most keys are exhausted")
    print("   1. Add more API keys to .env")
    print("   2. Consider upgrading to paid tier")
else:
    print("   ✅ You have enough working keys")
    print("   Your app should work now!")

print("\n" + "="*60)
