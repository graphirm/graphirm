use std::collections::HashMap;

use crate::ToolError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Allow,
    Ask,
    Deny,
}

pub struct ToolPermissions {
    defaults: HashMap<String, Permission>,
    fallback: Permission,
}

impl ToolPermissions {
    pub fn new(fallback: Permission) -> Self {
        Self { defaults: HashMap::new(), fallback }
    }

    pub fn allow_all() -> Self {
        Self::new(Permission::Allow)
    }

    pub fn ask_all() -> Self {
        Self::new(Permission::Ask)
    }

    pub fn set(&mut self, tool_name: impl Into<String>, permission: Permission) {
        self.defaults.insert(tool_name.into(), permission);
    }

    pub fn get(&self, tool_name: &str) -> Permission {
        self.defaults.get(tool_name).copied().unwrap_or(self.fallback)
    }

    pub fn check(&self, tool_name: &str) -> Result<Permission, ToolError> {
        match self.get(tool_name) {
            Permission::Deny => Err(ToolError::PermissionDenied(format!(
                "tool '{}' is denied by policy",
                tool_name
            ))),
            perm => Ok(perm),
        }
    }
}

impl Default for ToolPermissions {
    fn default() -> Self {
        Self::ask_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_all_returns_allow() {
        let perms = ToolPermissions::allow_all();
        assert_eq!(perms.get("bash"), Permission::Allow);
        assert_eq!(perms.get("write"), Permission::Allow);
    }

    #[test]
    fn ask_all_returns_ask() {
        let perms = ToolPermissions::ask_all();
        assert_eq!(perms.get("bash"), Permission::Ask);
    }

    #[test]
    fn default_is_ask_all() {
        let perms = ToolPermissions::default();
        assert_eq!(perms.get("any_tool"), Permission::Ask);
    }

    #[test]
    fn set_overrides_fallback() {
        let mut perms = ToolPermissions::ask_all();
        perms.set("bash", Permission::Allow);
        assert_eq!(perms.get("bash"), Permission::Allow);
        assert_eq!(perms.get("write"), Permission::Ask);
    }

    #[test]
    fn check_deny_returns_err() {
        let mut perms = ToolPermissions::allow_all();
        perms.set("rm", Permission::Deny);
        let result = perms.check("rm");
        assert!(matches!(result, Err(ToolError::PermissionDenied(_))));
    }

    #[test]
    fn check_allow_returns_ok() {
        let perms = ToolPermissions::allow_all();
        let result = perms.check("bash");
        assert!(matches!(result, Ok(Permission::Allow)));
    }

    #[test]
    fn check_ask_returns_ok_ask() {
        let perms = ToolPermissions::ask_all();
        let result = perms.check("write");
        assert!(matches!(result, Ok(Permission::Ask)));
    }
}
