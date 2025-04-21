# utils.py - PatchDB-Specific Prompt Templates

# **System Instruction**
SYS_INST = "You are a security expert that is good at static program analysis."

# **Standard Classification Prompt**
PROMPT_INST = """Please analyze the following code:
```
{diff_code}
```
Please indicate your analysis result with one of the options: 
(1) YES: This patch fixes a security vulnerability.
(2) NO: This patch does NOT fix a security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""

# **Zero-Shot Learning Example**
ZERO_SHOT_PROMPT = """Please analyze the following patch:

```
{diff_code}
```
Does this patch fix a security vulnerability? Reply with:
(1) YES: This patch fixes a security vulnerability.
(2) NO: This patch does NOT fix a security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""



# **One-Shot Learning Example**
ONESHOT_USER = """Please analyze the following patch:
 ```
diff --git a/src/firejail/fs_home.c b/src/firejail/fs_home.c
index d8ff636a9..2d19c8e94 100644
--- a/src/firejail/fs_home.c
+++ b/src/firejail/fs_home.c
@@ -42,8 +42,12 @@ static void skel(const char *homedir, uid_t u, gid_t g) {
              // don't copy it if we already have the file
              if (stat(fname, &s) == 0)
                      return;
+               if (is_link(fname)) { // stat on dangling symlinks fails, try again using lstat
+                       fprintf(stderr, "Error: invalid %s file\n", fname);
+                       exit(1);
+               }
              if (stat("/etc/skel/.zshrc", &s) == 0) {
-                       copy_file("/etc/skel/.zshrc", fname, u, g, 0644);
+                       copy_file_as_user("/etc/skel/.zshrc", fname, u, g, 0644);
                      fs_logger("clone /etc/skel/.zshrc");
              }
              else {
        ```
Please indicate your analysis result with one of the options: 
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability. 

Only reply with one of the options above. Do not include any further information.
"""

ONESHOT_ASSISTANT = "(1) YES: A security vulnerability detected."

# **Two-Shot Learning Example**
TWOSHOT_USER = """Please analyze the following patch:
```
```
diff --git a/include/Flow.h b/include/Flow.h
index f819f4b0..a8dac48d 100644
--- a/include/Flow.h
+++ b/include/Flow.h
@@ -236,8 +236,10 @@ class Flow : public GenericHashEntry {
   inline u_int8_t getTcpFlags()        { return(src2dst_tcp_flags | dst2src_tcp_flags);  };
   inline u_int8_t getTcpFlagsCli2Srv() { return(src2dst_tcp_flags);                      };
   inline u_int8_t getTcpFlagsSrv2Cli() { return(dst2src_tcp_flags);                      };
+#ifdef NTOPNG_PRO
   bool isPassVerdict();
   void setDropVerdict()         { passVerdict = false; };
+#endif
```
Please indicate your analysis result with one of the options: 
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability. 

Only reply with one of the options above. Do not include any further information.
"""

TWOSHOT_ASSISTANT = "(2) NO: No security vulnerability."

# **Chain of Thought Prompt**
CoT_PROMPT_TEMPLATE = """
Please analyze the following code patch:

```
{diff_code}
```

Let's analyze step-by-step:

1. **Context Understanding:** Understand the purpose of this code and its role in the system.
2. **Potential Issue Identification:** Determine whether the original code contains a security vulnerability. If so, identify its type (e.g., buffer overflow, injection, privilege escalation).
3. **Patch Evaluation:** Understand how the patch modifies the code and whether it mitigates the identified security issue.
4. **Effectiveness Assessment:** Understand and evaluate whether the patch fully resolves the vulnerability or introduces new risks.
5. **Final Decision:**
   (1) YES: This patch fixes a security vulnerability.
   (2) NO: This patch does NOT fix a security vulnerability.

Only provide an answer to step 5. Do not include any further explanation.
"""


# **Semantic Code Prompt**
SEMANTIC_CODE_PROMPT = """Please analyze the following patch:

### **Code Patch:**
```
{diff_code}
```

### **Semantic Information (Structured Analysis of the Code Changes):**
The following metadata provides a structured breakdown of the modifications in this patch. 

- **Added Function Calls:** 
```
{added_function_calls}
```

- **Removed Function Calls:** 

```
{removed_function_calls}
```

- **Added Variables:**  
```
{added_variables}
```

- **Removed Variables:** 
```
{removed_variables}
```

- **Added Control Structures:** 
```
{added_control_structures}
```

- **Removed Control Structures:** 
```
{removed_control_structures}
```
Please indicate your analysis result with one of the options: 
(1) YES: This patch fixes a security vulnerability.
(2) NO: This patch does NOT fix a security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""
