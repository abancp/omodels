/**
 * Model Registry — the core abstraction that makes the system modular.
 */
/* ─── Registry ─── */
const registry = new Map();
const categories = new Map();
export function registerModel(descriptor) {
    registry.set(descriptor.id, descriptor);
    if (!categories.has(descriptor.category)) {
        categories.set(descriptor.category, {
            id: descriptor.category,
            name: descriptor.category,
            icon: descriptor.categoryIcon,
        });
    }
}
export function getModel(id) {
    return registry.get(id);
}
export function getAllModels() {
    return Array.from(registry.values());
}
export function getCategories() {
    return Array.from(categories.values());
}
export function getModelsByCategory(categoryId) {
    return getAllModels().filter((m) => m.category === categoryId);
}
