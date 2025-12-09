# Security Notice

## Known Vulnerabilities

### MLflow Unsafe Deserialization (Unpatched)

**Status**: ⚠️ Known Issue - No patch available

**Affected Dependency**: mlflow (all versions 0.5.0 to 3.4.0)

**Current Version**: 2.22.4 (latest stable as of December 2024)

**Vulnerability**: Unsafe deserialization vulnerability in MLflow

**Risk Level**: Medium to High (depending on usage)

**CVE/Advisory**: Multiple CVEs for unsafe deserialization in MLflow

---

## Impact Assessment

### What is Affected?

MLflow is used in **Phase 2 (Supervised Learning)** for experiment tracking:
- Experiment logging
- Model artifact storage
- Metrics tracking
- Parameter logging

### Risk Factors

**HIGH RISK scenarios:**
- Loading MLflow models from untrusted sources
- Deserializing model artifacts from unknown origins
- Running MLflow tracking server exposed to public networks
- Processing user-uploaded model files

**LOW RISK scenarios (current project usage):**
- Using MLflow only for local experiment tracking
- Loading only self-generated models
- No external model inputs
- Running in isolated development environment

---

## Mitigation Strategies

### For Development/Research Use

If you're using this project for research or development with trusted data:

1. **✅ SAFE**: Use MLflow for local experiment tracking
   ```bash
   # This is safe for local use
   python phase2_supervised_learning.py
   mlflow ui --host 127.0.0.1  # Only localhost
   ```

2. **✅ SAFE**: Load only models you created yourself
   ```python
   # Safe - loading your own trained models
   model = mlflow.pyfunc.load_model("models/my_model")
   ```

3. **⚠️ AVOID**: Don't deserialize models from untrusted sources
   ```python
   # DANGEROUS - don't do this with untrusted sources
   model = mlflow.pyfunc.load_model("untrusted_model_url")
   ```

### For Production Deployment

**RECOMMENDED**: Consider alternatives for production:

1. **Option 1: Skip MLflow in Production**
   - Use MLflow only in development
   - Export models to standard formats (ONNX, pickle, joblib)
   - Deploy models without MLflow runtime

   ```python
   # Development: Train with MLflow
   with mlflow.start_run():
       model.fit(X, y)
       mlflow.sklearn.log_model(model, "model")
   
   # Production: Export without MLflow
   import joblib
   joblib.dump(model, "model.pkl")
   ```

2. **Option 2: Network Isolation**
   - Run MLflow in isolated network segments
   - Use firewall rules to restrict access
   - Only allow connections from trusted sources
   - Never expose MLflow tracking server to public internet

3. **Option 3: Input Validation**
   - Implement strict validation for model inputs
   - Use checksums/signatures for model files
   - Only load models from verified sources
   - Maintain an allowlist of trusted model sources

4. **Option 4: Containerization**
   - Run MLflow in isolated containers
   - Use read-only filesystems where possible
   - Implement least-privilege access controls
   - Monitor for suspicious deserialization activity

---

## Security Best Practices

### General Guidelines

1. **Never expose MLflow tracking server to public internet**
   ```bash
   # GOOD - localhost only
   mlflow ui --host 127.0.0.1 --port 5000
   
   # BAD - accessible from anywhere
   mlflow ui --host 0.0.0.0 --port 5000
   ```

2. **Don't load models from untrusted sources**
   - Only load models you created
   - Verify model integrity before loading
   - Use checksums or digital signatures

3. **Keep MLflow isolated**
   - Use separate environments for MLflow
   - Don't run MLflow with elevated privileges
   - Limit network access to MLflow services

4. **Monitor for updates**
   - Watch MLflow GitHub repository for security patches
   - Subscribe to MLflow security advisories
   - Update when patches become available

### For This Project Specifically

**Current Usage**: MLflow is optional and used only for Phase 2 experiment tracking

**Risk Level**: **LOW** for typical usage scenarios because:
- ✅ MLflow is optional (you can skip Phase 2 or use without MLflow)
- ✅ No external model loading in default workflow
- ✅ No public-facing MLflow services
- ✅ All models are self-generated from trusted code

**Recommendation**: Safe to use for research/development, but follow production guidelines above for deployment

---

## Alternative Experiment Tracking Tools

If the MLflow vulnerability is a concern, consider these alternatives:

1. **Weights & Biases (wandb)**
   - Cloud-based experiment tracking
   - No local deserialization issues
   - Free tier available

2. **TensorBoard**
   - Lighter weight
   - Visualization focused
   - Part of TensorFlow ecosystem

3. **Neptune.ai**
   - Cloud-based
   - Good security posture
   - Team collaboration features

4. **Custom Logging**
   - Use JSON/CSV for metrics
   - Simple and secure
   - Full control

---

## Disabling MLflow

If you want to completely avoid MLflow:

### Option 1: Skip MLflow Installation

```bash
# Install without MLflow
pip install pandas numpy scikit-learn scipy joblib matplotlib seaborn xgboost
```

### Option 2: Modify Phase 2 Script

Comment out MLflow imports in `phase2_supervised_learning.py`:

```python
# import mlflow  # Commented out
# mlflow.start_run()  # Skip MLflow logging
```

The experiments will still run and save models, just without MLflow tracking.

---

## Monitoring for Patches

**Check for updates regularly:**

1. **MLflow GitHub**: https://github.com/mlflow/mlflow
2. **MLflow Security Advisories**: https://github.com/mlflow/mlflow/security/advisories
3. **CVE Database**: Search for "MLflow" at https://cve.mitre.org

**When a patch is released:**

1. Update `requirements.txt`:
   ```bash
   mlflow==<patched-version>
   ```

2. Update `INSTALLATION.md` with new version

3. Test Phase 2 to ensure compatibility

4. Update this SECURITY.md document

---

## Reporting Security Issues

If you discover security issues in this project:

1. **Do NOT** open public GitHub issues for security vulnerabilities
2. Contact the project maintainers directly
3. Provide detailed information about the vulnerability
4. Allow time for assessment and patching

---

## Version History

| Date | MLflow Version | Status | Notes |
|------|---------------|--------|-------|
| Dec 2024 | 2.9.1 | ❌ Vulnerable | 33+ known vulnerabilities |
| Dec 2024 | 2.22.4 | ⚠️ Partially Fixed | Most vulnerabilities patched, unsafe deserialization remains |

---

## Disclaimer

This security notice is provided for informational purposes. Users are responsible for their own security assessments and risk management. The project maintainers make no warranties regarding the security of dependencies.

**Use at your own risk in production environments.**

---

## Summary

- ✅ **For Research/Development**: Safe to use with local, trusted data
- ⚠️ **For Production**: Follow mitigation strategies or use alternatives
- 🔄 **Status**: Monitoring for patches from MLflow maintainers
- 📋 **Action**: Document acknowledged, risk accepted for development use

**Last Updated**: December 2024
**Next Review**: When MLflow releases security patches
