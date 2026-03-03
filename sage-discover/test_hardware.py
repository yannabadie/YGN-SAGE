import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../sage-python/src"))

import sage_core

def main():
    print("🖥️  YGN-SAGE SOTA Hardware Auto-Discovery (Phase 2) 🖥️")
    print("=" * 60)
    
    try:
        hw = sage_core.HardwareProfile.detect()
        print(f"Processor: {hw.cpu_brand}")
        print(f"Cores: {hw.physical_cores} Physical / {hw.logical_cores} Logical")
        print(f"Memory: {hw.free_memory_mb} MB Free / {hw.total_memory_mb} MB Total")
        
        print("\n⚡ ASI Vectorization Capabilities:")
        print(f"  - AVX2: {'✅ Detected' if hw.has_avx2 else '❌ Not Found'}")
        print(f"  - AVX-512: {'✅ Detected' if hw.has_avx512 else '❌ Not Found'}")
        print(f"  - ARM NEON: {'✅ Detected' if hw.has_neon else '❌ Not Found'}")
        
        if hw.is_simd_capable:
            print("\n🚀 SOTA SYSTEM READY: SIMD Vectorization is available for contiguous memory graphs.")
        else:
            print("\n⚠️ WARNING: System lacks SIMD instructions. GraphRAG traversal will not be vectorized.")
            
    except Exception as e:
        print(f"❌ Error detecting hardware: {e}")

if __name__ == "__main__":
    main()
